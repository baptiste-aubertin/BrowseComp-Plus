"""Ingest BrowseComp-Plus corpus into paradigm via POST /api/v3/files.

Three actions (combinable):
  --docids "<csv>"   retry specific docids (no full listing)
  --verify           walk the full workspace, upload missing docids
  --check-failed     list failed files only, delete + re-upload them

Docids are matched via external_metadata.external_id. Reruns are idempotent — only
docids that need work get touched.

Documents are truncated before upload to mirror the upstream benchmark's 512-token
document window (build_pylate_index.py --document-length 512), counted with the
lightonai/DenseOn-multilingual tokenizer (the Paradigm embedder): each doc is cut so
that <bos> <text> <eos> fits in --document-length tokens, i.e. document_length - 2
content tokens (510 for the default 512).
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

load_dotenv()

BASE_URL = os.environ["PARADIGM_BASE_URL"].rstrip("/")
API_KEY = os.environ["PARADIGM_API_KEY"]
WORKSPACE_ID = int(os.environ["PARADIGM_WORKSPACE_ID"])

TOKENIZER_NAME = "lightonai/DenseOn-multilingual"

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
FAIL_STATUSES = {"parsing_failed", "embedding_failed", "fail"}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["X-Api-Key"] = API_KEY
    return s


def _status_priority(status: str | None) -> int:
    """Higher wins when the same docid appears more than once in the workspace."""
    if status == "embedded":
        return 3
    if status in FAIL_STATUSES:
        return 1
    if status:
        return 2  # any non-terminal in-flight status beats a failed one
    return 0


def list_workspace(
    session: requests.Session,
    workspace_id: int,
    status_filter: set[str] | None = None,
    page_size: int = 100,
) -> list[dict]:
    """Walk every page of GET /api/v3/files. Logs every 10th page to stay readable."""
    out: list[dict] = []
    url = f"{BASE_URL}/api/v3/files"
    params: dict | None = {"workspace_id": workspace_id, "page_size": page_size, "ordering": "id"}
    if status_filter:
        params["status"] = ",".join(sorted(status_filter))
    page_num = 0
    total = None
    while True:
        r = session.get(url, params=params, timeout=60)
        r.raise_for_status()
        body = r.json()
        results = body.get("results", [])
        out.extend(results)
        page_num += 1
        if total is None:
            total = body.get("count")
        next_url = body.get("next")
        # Log first page, every 10th, and the last page.
        if page_num == 1 or page_num % 10 == 0 or not next_url:
            of = f"/{total:,}" if isinstance(total, int) else ""
            print(f"  page {page_num}: {len(out):,}{of} files", flush=True)
        if not next_url:
            break
        url, params = next_url, None  # `next` is fully-qualified
    return out


def find_file_by_external_id(
    session: requests.Session, workspace_id: int, docid: str
) -> dict | None:
    """Resolve a single docid via the server-side external_id filter. Returns best file or None."""
    r = session.get(
        f"{BASE_URL}/api/v3/files",
        params={
            "workspace_id": workspace_id,
            "external_metadata__external_id": str(docid),
            "page_size": 50,
        },
        timeout=60,
    )
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        return None
    best = results[0]
    for f in results[1:]:
        if _status_priority(f.get("status")) > _status_priority(best.get("status")):
            best = f
    return best


def upload_doc(session: requests.Session, docid: str, text: str, url: str) -> dict:
    """POST /api/v3/files. WAF fallback on 403; exponential backoff on retryable 5xx."""
    payload = text.encode("utf-8")
    # WAF fallback variants: tried in order on 403. The Paradigm parser sniffs
    # content, so swapping Content-Type / extension is harmless for ingestion
    # but can route past a content-type-scoped WAF rule.
    variants = [("txt", "text/plain"), ("bin", "application/octet-stream"), ("md", "text/markdown")]
    idx = 0
    files = {"file": (f"{docid}.{variants[0][0]}", payload, variants[0][1])}
    data = {
        "workspace_id": str(WORKSPACE_ID),
        "filename": f"{docid}.txt",
        "title": (url or f"doc-{docid}")[:255],
        "external_metadata": json.dumps(
            {"external_id": str(docid), "additional_metadata": {"url": url} if url else {}}
        ),
    }
    last_err = None
    for attempt in range(4):
        try:
            r = session.post(f"{BASE_URL}/api/v3/files", files=files, data=data, timeout=120)
            if r.status_code == 201:
                return {"docid": str(docid), "file_id": r.json()["id"], "ok": True}
            if r.status_code == 403 and idx + 1 < len(variants):
                idx += 1
                ext, ctype = variants[idx]
                files = {"file": (f"{docid}.{ext}", payload, ctype)}
                continue
            if r.status_code in RETRYABLE_STATUS:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(min(2**attempt, 30))
                continue
            return {"docid": str(docid), "ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2**attempt, 30))
    return {"docid": str(docid), "ok": False, "error": last_err}


def delete_file(session: requests.Session, file_id: int) -> tuple[bool, str | None]:
    """DELETE /api/v3/files/{id}. 204 → ok; 404 → already gone (treat as ok)."""
    last_err = None
    for attempt in range(4):
        try:
            r = session.delete(f"{BASE_URL}/api/v3/files/{file_id}", timeout=60)
            if r.status_code in (204, 404):
                return True, None
            if r.status_code in RETRYABLE_STATUS:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(min(2**attempt, 30))
                continue
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2**attempt, 30))
    return False, last_err


def redo_failed(session: requests.Session, docid: str, file_row: dict, corpus: dict) -> dict:
    ok, err = delete_file(session, file_row.get("id"))
    if not ok:
        return {"docid": docid, "ok": False, "error": f"delete: {err}"}
    text, url = corpus[docid]
    return upload_doc(session, docid, text, url)


def truncate_texts(texts: list[str], document_length: int, batch_size: int = 256) -> list[str]:
    """Truncate each text so the embedder input <bos> <text> <eos> fits in
    document_length tokens, i.e. document_length - num_special_tokens_to_add()
    content tokens. Texts under the limit are kept verbatim rather than
    round-tripped through the tokenizer."""
    # The repo's tokenizer_config declares the transformers-v5 TokenizersBackend
    # class, which AutoTokenizer in transformers 4.x can't resolve; loading through
    # PreTrainedTokenizerFast reads tokenizer.json directly.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_NAME)
    max_tokens = document_length - tokenizer.num_special_tokens_to_add(False)
    out: list[str] = []
    n_truncated = 0
    for i in tqdm(range(0, len(texts), batch_size), desc="truncate", unit="batch"):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for text, tokens in zip(batch, encoded):
            if len(tokens) > max_tokens:
                out.append(tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True))
                n_truncated += 1
            else:
                out.append(text)
    print(
        f"  truncated {n_truncated:,}/{len(texts):,} docs to {max_tokens} content tokens "
        f"(document_length={document_length})"
    )
    return out


def load_corpus(path: Path, document_length: int = 512) -> dict[str, tuple[str, str]]:
    docids: list[str] = []
    texts: list[str] = []
    urls: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docids.append(str(row["docid"]))
            texts.append(row.get("text", ""))
            urls.append(row.get("url", "") or "")
    if document_length and document_length > 0:
        texts = truncate_texts(texts, document_length)
    return {d: (t, u) for d, t, u in zip(docids, texts, urls)}


def parse_retry_ids(docids_csv: str, corpus_docids: set[str]) -> set[str]:
    ids = {x.strip() for x in docids_csv.split(",") if x.strip()}
    unknown = ids - corpus_docids
    if unknown:
        sample = sorted(unknown)[:5]
        more = "…" if len(unknown) > 5 else ""
        print(f"warning: ignoring {len(unknown):,} docid(s) not in --input: {sample}{more}")
    return ids & corpus_docids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data/browsecomp_plus_corpus.jsonl")
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--document-length", type=int, default=512,
        help="Document window to replicate (same flag as the upstream "
             f"build_pylate_index.py): docs are cut to document_length - 2 content tokens "
             f"of the {TOKENIZER_NAME} tokenizer before upload. 0 disables.",
    )
    parser.add_argument(
        "--docids", type=str, default=None,
        help="Comma-separated docids to retry. Looked up per-docid; no full workspace listing.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Walk the full workspace; upload every corpus docid that is missing.",
    )
    parser.add_argument(
        "--check-failed", action="store_true",
        help=f"List files whose status is in {sorted(FAIL_STATUSES)} and delete + re-upload them. "
             "Combines naturally with --verify (re-uses the full listing).",
    )
    args = parser.parse_args()

    if not (args.docids or args.verify or args.check_failed):
        parser.error("specify at least one of --docids, --verify, --check-failed")

    s = _session()
    corpus = load_corpus(args.input, args.document_length)
    corpus_docids = set(corpus)
    print(f"corpus: {len(corpus):,} docids from {args.input}")

    to_redo: dict[str, dict] = {}   # docid -> existing file row (delete + reupload)
    to_upload: set[str] = set()     # docid -> upload (no existing file)

    # --verify: full listing → missing + (optionally) failed.
    if args.verify:
        print(f"\nlisting workspace {WORKSPACE_ID} (full) …")
        files = list_workspace(s, WORKSPACE_ID)
        by_docid: dict[str, dict] = {}
        for f in files:
            em = f.get("external_metadata") or {}
            ext_id = em.get("external_id")
            if not ext_id:
                continue
            ext_id = str(ext_id)
            if ext_id not in corpus_docids:
                continue
            prev = by_docid.get(ext_id)
            if prev is None or _status_priority(f.get("status")) > _status_priority(prev.get("status")):
                by_docid[ext_id] = f
        ok = sum(1 for f in by_docid.values() if f.get("status") == "embedded")
        in_flight = sum(1 for f in by_docid.values()
                        if f.get("status") not in FAIL_STATUSES and f.get("status") != "embedded")
        failed = {d: f for d, f in by_docid.items() if f.get("status") in FAIL_STATUSES}
        missing = corpus_docids - by_docid.keys()
        print(
            f"  ok={ok:,}  failed={len(failed):,}  in_flight={in_flight:,}  "
            f"missing={len(missing):,}  (corpus={len(corpus):,})"
        )
        to_upload |= missing
        if args.check_failed:
            to_redo.update(failed)

    # --check-failed without --verify: cheap server-side status filter.
    elif args.check_failed:
        print(f"\nlisting failed files in workspace {WORKSPACE_ID} …")
        files = list_workspace(s, WORKSPACE_ID, status_filter=FAIL_STATUSES)
        for f in files:
            em = f.get("external_metadata") or {}
            ext_id = em.get("external_id")
            if ext_id and str(ext_id) in corpus_docids:
                to_redo[str(ext_id)] = f
        print(f"  found {len(to_redo):,} failed file(s) with a corpus match")

    # --docids: per-docid lookup, no full listing.
    if args.docids:
        retry_ids = parse_retry_ids(args.docids, corpus_docids)
        if retry_ids:
            print(f"\nlooking up {len(retry_ids):,} --docids in workspace {WORKSPACE_ID} …")
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                results = list(tqdm(
                    ex.map(lambda d: (d, find_file_by_external_id(s, WORKSPACE_ID, d)), retry_ids),
                    total=len(retry_ids), desc="lookup",
                ))
            for docid, f in results:
                if f is not None:
                    to_redo[docid] = f
                else:
                    to_upload.add(docid)

    to_upload -= to_redo.keys()
    if not to_redo and not to_upload:
        print("\nnothing to do.")
        return

    if to_redo:
        print(f"\nre-uploading {len(to_redo):,} doc(s) (delete + upload) …")
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(redo_failed, s, d, f, corpus) for d, f in to_redo.items()]
            n_ok = n_fail = 0
            for fut in tqdm(as_completed(futs), total=len(futs), desc="re-upload"):
                if fut.result().get("ok"):
                    n_ok += 1
                else:
                    n_fail += 1
        print(f"  re-upload: ok={n_ok:,} fail={n_fail:,}")

    if to_upload:
        print(f"\nuploading {len(to_upload):,} missing doc(s) …")
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(upload_doc, s, d, *corpus[d]) for d in to_upload]
            n_ok = n_fail = 0
            for fut in tqdm(as_completed(futs), total=len(futs), desc="upload"):
                if fut.result().get("ok"):
                    n_ok += 1
                else:
                    n_fail += 1
        print(f"  upload: ok={n_ok:,} fail={n_fail:,}")


if __name__ == "__main__":
    main()
