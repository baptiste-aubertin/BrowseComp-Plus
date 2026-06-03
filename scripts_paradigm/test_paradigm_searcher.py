"""Smoke test for ParadigmSearcher against .env credentials."""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def _load_paradigm_searcher():
    """Load ParadigmSearcher without triggering searchers/__init__.py (pyserini needs Java)."""
    pkg_dir = ROOT / "searcher" / "searchers"
    pkg_name = "_paradigm_pkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(pkg_dir)]
    sys.modules[pkg_name] = pkg
    for sub in ("base", "paradigm_searcher"):
        spec = importlib.util.spec_from_file_location(f"{pkg_name}.{sub}", pkg_dir / f"{sub}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{pkg_name}.{sub}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules[f"{pkg_name}.paradigm_searcher"].ParadigmSearcher


ParadigmSearcher = _load_paradigm_searcher()


def main() -> None:
    parser = argparse.ArgumentParser()
    ParadigmSearcher.parse_args(parser)
    parser.add_argument("--query", default="search doc about cars")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fetch-doc", action="store_true", help="Also call get_document on the top hit.")
    args = parser.parse_args()

    searcher = ParadigmSearcher(args)

    print(f"\n=== search(query={args.query!r}, k={args.k}) ===")
    try:
        hits = searcher.search(args.query, k=args.k)
    except Exception as e:  # noqa: BLE001
        import requests
        if isinstance(e, requests.HTTPError) and e.response is not None:
            print(f"  HTTP {e.response.status_code}: {e.response.text[:500]}")
        raise
    print(f"got {len(hits)} hit(s)\n")
    for i, hit in enumerate(hits, 1):
        snippet = (hit.get("text") or "").replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "…"
        print(f"  [{i}] docid={hit['docid']}  score={hit['score']:.4f}")
        print(f"      {snippet}")

    if args.fetch_doc and hits:
        top_docid = hits[0]["docid"]
        print(f"\n=== get_document(docid={top_docid!r}) ===")
        doc = searcher.get_document(top_docid)
        if doc is None:
            print("  not found")
        else:
            text = doc.get("text") or ""
            print(f"  docid={doc['docid']}  text_chars={len(text)}")
            print(f"  preview: {text[:400]!r}{'…' if len(text) > 400 else ''}")


if __name__ == "__main__":
    main()
