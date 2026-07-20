"""LightOn Paradigm Console searcher: POST /api/v3/search + GET /api/v3/files/{id}."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from .base import BaseSearcher

# Load PARADIGM_* from the repo-root .env on import, before parse_args reads the
# env-var defaults below. Existing environment variables take precedence.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

logger = logging.getLogger(__name__)

# Chunks requested per /api/v3/search call. Larger than the k=5 docs handed to
# the LLM: the engine's internal fusion (dense + BM25 + ColBERT rerank +
# cross-encoder) operates on this candidate pool, so a wider request improves
# the final top-5. 50 was chosen over 100 after an offline eval on 250 labelled
# pairs showed identical gold-in-top-5 (80.4%) at half the cross-encoder cost.
PARADIGM_MAX_RESULTS = int(os.getenv("PARADIGM_MAX_RESULTS", "50"))
EXPAND_QUERIES = os.getenv("PARADIGM_EXPAND_QUERIES", "") == "1"
EXPAND_MODEL = os.getenv("PARADIGM_EXPAND_MODEL", "gpt-oss-120b")
EXPAND_MODEL_URL = os.getenv("PARADIGM_EXPAND_MODEL_URL", "http://87.120.213.41:8003/v1")

# Cross-encoder (reranker) modes for /api/v3/search
# (paradigm-mission-control #3685 / #3745):
# - scoring_and_filtering: score candidates, return only those above the quality
#   threshold; when none clears it, an adaptive fallback returns the few best
#   instead of an empty result. API default.
# - scoring_only: score every candidate and return them all (no threshold filter);
#   every chunk carries scores.relevance.
# - none: skip relevance scoring entirely (scores.relevance is null). Fastest.
RELEVANCE_SCORING_CHOICES = ["scoring_and_filtering", "scoring_only", "none"]
DEFAULT_RELEVANCE_SCORING = "scoring_and_filtering"

# Retry tuning. Transient failures (429 rate limits, connection errors, and 5xx
# responses) are all retried indefinitely until the request passes. Backoff is
# uncapped exponential, honoring a Retry-After header when the server sends one.
PARADIGM_BACKOFF_BASE = 1.0


class ParadigmSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument(
            "--base-url",
            default=os.getenv("PARADIGM_BASE_URL"),
            help="Paradigm Console base URL (default: $PARADIGM_BASE_URL).",
        )
        parser.add_argument(
            "--api-key",
            default=os.getenv("PARADIGM_API_KEY"),
            help="Paradigm API key (default: $PARADIGM_API_KEY).",
        )
        parser.add_argument(
            "--workspace-id",
            type=int,
            default=int(os.environ["PARADIGM_WORKSPACE_ID"])
            if os.getenv("PARADIGM_WORKSPACE_ID")
            else None,
            help="Restrict search to this workspace (default: $PARADIGM_WORKSPACE_ID, else cross-workspace).",
        )
        parser.add_argument(
            "--mode",
            choices=["text", "vision"],
            default="text",
            help="Search mode: 'text' (hybrid) or 'vision' (VLM page-image). Default: text.",
        )
        parser.add_argument(
            "--relevance-scoring",
            choices=RELEVANCE_SCORING_CHOICES,
            default=DEFAULT_RELEVANCE_SCORING,
            help=(
                "Cross-encoder relevance scoring mode: 'scoring_and_filtering' "
                "(default; threshold-filtered with adaptive fallback on empty), "
                "'scoring_only' (score all candidates, no filtering), or 'none' "
                "(skip scoring, scores.relevance null; fastest)."
            ),
        )
        parser.add_argument(
            "--corpus-path",
            default=os.getenv("BROWSECOMP_CORPUS_PATH", "data/browsecomp_plus_corpus.jsonl"),
            help="Local corpus JSONL with FULL document texts; get_document serves from here "
                 "(Paradigm only stores the 512-token ingestion window).",
        )
        parser.add_argument(
            "--request-timeout",
            type=float,
            default=120.0,
            help="HTTP timeout in seconds for /search and /files calls (default: 120).",
        )

    def __init__(self, args):
        self.args = args

        if not args.base_url:
            raise ValueError("paradigm --base-url is required (or set $PARADIGM_BASE_URL)")
        if not args.api_key:
            raise ValueError("paradigm --api-key is required (or set $PARADIGM_API_KEY)")

        self.base_url = args.base_url.rstrip("/")
        self.workspace_id = args.workspace_id
        self.mode = args.mode
        self.relevance_scoring = args.relevance_scoring
        self.timeout = args.request_timeout
        self.corpus_path = args.corpus_path
        self._corpus_index = None

        self.session = requests.Session()
        self.session.headers["X-Api-Key"] = args.api_key

        # docid (external_id) -> file_id, populated by search results and on-demand lookups.
        self._docid_to_file_id: Dict[str, int] = {}

        self._expand_cache: Dict[str, List[str]] = {}
        if EXPAND_QUERIES:
            import openai

            self._expand_client = openai.OpenAI(
                base_url=EXPAND_MODEL_URL,
                api_key=os.getenv("LLM_API_KEY", "EMPTY"),
                timeout=180,
            )

        logger.info(
            "Paradigm searcher ready (base_url=%s workspace_id=%s mode=%s relevance_scoring=%s)",
            self.base_url,
            self.workspace_id,
            self.mode,
            self.relevance_scoring,
        )

    @staticmethod
    def _retry_after_seconds(response: requests.Response, attempt: int) -> float:
        """Seconds to wait before retrying: Retry-After header if present, else exponential backoff."""
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return PARADIGM_BACKOFF_BASE * (2 ** attempt)

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Issue an HTTP request, retrying transient failures indefinitely.

        Connection errors, 429 rate limits, and 5xx responses are retried forever
        until the request succeeds (429 honors a Retry-After header when present).
        Other 4xx responses are deterministic and returned to the caller as-is.
        """
        kwargs.setdefault("timeout", self.timeout)
        transient_attempts = 0
        rate_limit_attempts = 0
        while True:
            try:
                response = self.session.request(method, url, **kwargs)
            except requests.RequestException as exc:
                wait = PARADIGM_BACKOFF_BASE * (2 ** transient_attempts)
                transient_attempts += 1
                logger.warning(
                    "Paradigm request error (%s %s): %s; retry %d in %.1fs",
                    method, url, exc, transient_attempts, wait,
                )
                time.sleep(wait)
                continue

            if response.ok:
                return response

            if response.status_code == 429:
                wait = self._retry_after_seconds(response, rate_limit_attempts)
                rate_limit_attempts += 1
                logger.warning(
                    "Paradigm rate limited (429 on %s %s); retry %d in %.1fs",
                    method, url, rate_limit_attempts, wait,
                )
                time.sleep(wait)
                continue

            if 400 <= response.status_code < 500:
                # Client errors other than 429 are deterministic: retrying the same
                # payload cannot succeed. Log the body (e.g. 422 field errors) and
                # hand the response back to the caller.
                logger.error(
                    "Paradigm client error (%d on %s %s): %s",
                    response.status_code, method, url, response.text[:500],
                )
                return response

            # 5xx: retry indefinitely with uncapped exponential backoff.
            wait = PARADIGM_BACKOFF_BASE * (2 ** transient_attempts)
            transient_attempts += 1
            logger.warning(
                "Paradigm server error (%d on %s %s); retry %d in %.1fs",
                response.status_code, method, url, transient_attempts, wait,
            )
            time.sleep(wait)

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Top-k unique docs; optionally via multi-query expansion (RRF merge).

        PARADIGM_EXPAND_QUERIES=1 mirrors the proposed server-side expansion:
        an LLM generates a keyword variant and a paraphrase, the three queries
        run concurrently, and the per-query doc rankings are merged with
        reciprocal-rank fusion (k=60). Offline study on 830 questions: pool
        ceiling +13.5, evidence@50 +8.8 vs the single query.
        """
        if not EXPAND_QUERIES:
            return self._search_one(query, k)
        variants = [query] + self._expand_query(query)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(variants)) as ex:
            rankings = list(ex.map(lambda q: self._search_one(q, PARADIGM_MAX_RESULTS), variants))
        rrf: Dict[str, float] = {}
        best_hit: Dict[str, Dict[str, Any]] = {}
        for ranked in rankings:
            for i, hit in enumerate(ranked):
                d = hit["docid"]
                rrf[d] = rrf.get(d, 0.0) + 1.0 / (60 + i + 1)
                cur = best_hit.get(d)
                if cur is None or hit["score"] > cur["score"]:
                    best_hit[d] = hit
        merged = sorted(rrf, key=lambda d: rrf[d], reverse=True)[:k]
        return [{**best_hit[d], "score": round(rrf[d], 6)} for d in merged]

    def _expand_query(self, query: str) -> List[str]:
        """Two LLM-generated variants (keyword, paraphrase); cached per query."""
        cached = self._expand_cache.get(query.strip().lower())
        if cached is not None:
            return cached
        prompt = (
            "Given a search query, produce exactly two alternative search queries for a "
            "keyword+semantic search engine:\n"
            "1. KEYWORD: the key entities, names, dates and rare terms only (no filler words).\n"
            "2. PARAPHRASE: a reformulation using different vocabulary/synonyms that preserves "
            "the information need.\n\n"
            'Reply as strict JSON: {"keyword": "...", "paraphrase": "..."}\n\n'
            f"Search query: {query}"
        )
        delay = 1.0
        for _ in range(6):
            try:
                resp = self._expand_client.chat.completions.create(
                    model=EXPAND_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=2048,
                )
                txt = resp.choices[0].message.content
                d = json.loads(txt[txt.find("{") : txt.rfind("}") + 1])
                out = [str(d["keyword"]).strip()[:1500], str(d["paraphrase"]).strip()[:1500]]
                out = [v for v in out if v]
                self._expand_cache[query.strip().lower()] = out
                return out
            except Exception as e:
                logger.warning("query expansion failed (%s); retrying", str(e)[:80])
                time.sleep(min(delay, 15))
                delay *= 2
        # Expansion is an enhancement — degrade to the plain query, never block search.
        self._expand_cache[query.strip().lower()] = []
        return []

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """Strip control characters LLMs occasionally emit (GPT-5 produced
        \x00 inside quoted search strings, which the API rejects with 422
        "Null characters are not allowed") — a lost search round otherwise."""
        return "".join(ch for ch in query if ch >= " " or ch in "\n\t").strip()

    def _search_one(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        query = self._sanitize_query(query)
        # Always pull the API's maximum candidate pool; the cross-encoder scores
        # exactly these max_results candidates (no overfetched tail since #3685).
        # Then dedup by docid (best chunk per doc) and take top-k unique docids.
        # Billing is one search credit regardless of max_results. Note that in
        # scoring_and_filtering mode the API may return fewer than max_results
        # chunks (threshold filtering), so fewer than k unique docs is possible;
        # use --relevance-scoring scoring_only to keep the full scored pool.
        payload: Dict[str, Any] = {
            "query": query,
            "max_results": PARADIGM_MAX_RESULTS,
            "mode": self.mode,
        }
        # Newer API versions only accept explicit non-default modes in the
        # request (choices: none / scoring_only); scoring_and_filtering is the
        # server default and must be expressed by omitting the field.
        if self.relevance_scoring != "scoring_and_filtering":
            payload["relevance_scoring"] = self.relevance_scoring
        if self.workspace_id is not None:
            payload["workspace_id"] = [self.workspace_id]

        r = self._request("POST", f"{self.base_url}/api/v3/search", json=payload)
        r.raise_for_status()
        results = r.json().get("results", [])

        # Keep the highest-scoring chunk per docid (explicit max, independent of response order),
        # then sort docids by best-chunk score and take top-k.
        best_per_docid: Dict[str, Dict[str, Any]] = {}
        for hit in results:
            source = hit.get("source") or {}
            ext_meta = source.get("external_metadata") or {}
            docid = ext_meta.get("external_id")
            if not docid:
                # Fall back to file_id when external_id is missing (e.g. directly-uploaded docs).
                file_id = source.get("file_id")
                if file_id is None:
                    continue
                docid = str(file_id)
            docid = str(docid)

            file_id = source.get("file_id")
            if file_id is not None:
                self._docid_to_file_id[docid] = int(file_id)

            # v3 schema: "score" is the fused float; "scores" holds the per-signal
            # breakdown, where "relevance" is the reranker confidence (null when
            # relevance_scoring="none").
            scores = hit.get("scores") or {}
            score_value = scores.get("relevance")
            if score_value is None:
                score_value = hit.get("score")
            score_value = float(score_value) if score_value is not None else 0.0

            existing = best_per_docid.get(docid)
            if existing is not None and existing["score"] >= score_value:
                continue

            best_per_docid[docid] = {
                "docid": docid,
                "score": score_value,
                "text": hit.get("content") or "",
            }

        ranked = sorted(best_per_docid.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:k]

    def _resolve_file_id(self, docid: str) -> Optional[int]:
        cached = self._docid_to_file_id.get(docid)
        if cached is not None:
            return cached

        params: Dict[str, Any] = {
            "external_metadata__external_id": docid,
            "page_size": 50,
        }
        if self.workspace_id is not None:
            params["workspace_id"] = self.workspace_id

        r = self._request("GET", f"{self.base_url}/api/v3/files", params=params)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None

        # Prefer an embedded file; otherwise take the first match.
        chosen = next((f for f in results if f.get("status") == "embedded"), results[0])
        file_id = int(chosen["id"])
        self._docid_to_file_id[docid] = file_id
        return file_id

    def _corpus_offsets(self) -> Dict[str, int]:
        """Lazy byte-offset index over the local corpus JSONL (docid -> offset).

        Paradigm stores only the 512-token ingestion window per document;
        ``get_document`` must return the *full* original text, exactly like the
        PyLate searcher does from the HF corpus. A seek index keeps memory flat
        instead of holding the whole ~3 GB corpus in RAM.
        """
        if getattr(self, "_corpus_index", None) is None:
            index: Dict[str, int] = {}
            offset = 0
            with open(self.corpus_path, "rb") as f:
                for line in f:
                    # docid is the first key of every corpus row — cheap parse.
                    key = line[: line.find(b",")].split(b":", 1)[-1].strip(b' "')
                    index[key.decode()] = offset
                    offset += len(line)
            logger.info("Indexed %d corpus documents from %s", len(index), self.corpus_path)
            self._corpus_index = index
        return self._corpus_index

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        offset = self._corpus_offsets().get(str(docid))
        if offset is None:
            return None
        with open(self.corpus_path, "rb") as f:
            f.seek(offset)
            row = json.loads(f.readline())
        return {"docid": str(row["docid"]), "text": row.get("text", "")}

    @property
    def search_type(self) -> str:
        return "Paradigm-Console"

    def search_description(self, k: int = 10) -> str:
        return (
            f"Perform a hybrid retrieval search against the LightOn Paradigm knowledge base. "
            f"Returns up to {k} unique documents with docid, score, and snippet "
            f"(the highest-scoring chunk's text content for each document)."
        )

    def get_document_description(self) -> str:
        return "Retrieve a full document by its docid."
