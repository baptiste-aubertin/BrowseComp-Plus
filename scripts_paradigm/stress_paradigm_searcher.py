"""Concurrent retrieval stress test for ParadigmSearcher against live .env creds.

Drives the real BrowseComp-Plus queries through search() (and optionally
get_document) on many threads to exercise the retry/429 handling under load.
"""

import argparse
import importlib.util
import logging
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def _load_paradigm_searcher():
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


def load_queries(tsv: Path, limit: int):
    queries = []
    with tsv.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, _, text = line.partition("\t")
            queries.append((qid, text or qid))
            if limit and len(queries) >= limit:
                break
    return queries


def main() -> None:
    parser = argparse.ArgumentParser()
    ParadigmSearcher.parse_args(parser)
    parser.add_argument("--query-file", default=str(ROOT / "topics-qrels" / "queries.tsv"))
    parser.add_argument("--limit", type=int, default=50, help="Number of queries (0 = all).")
    parser.add_argument("--num-threads", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fetch-doc", action="store_true", help="Also get_document the top hit.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    searcher = ParadigmSearcher(args)
    queries = load_queries(Path(args.query_file), args.limit)
    print(f"Loaded {len(queries)} queries; {args.num_threads} threads; k={args.k}")

    ok = err = 0
    rate_429 = 0
    t0 = time.monotonic()

    def run_one(item):
        qid, text = item
        hits = searcher.search(text, k=args.k)
        n_doc = 0
        if args.fetch_doc and hits:
            doc = searcher.get_document(hits[0]["docid"])
            n_doc = len(doc["text"]) if doc else 0
        return qid, len(hits), n_doc

    with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
        futures = {ex.submit(run_one, q): q for q in queries}
        for fut in as_completed(futures):
            qid, _, text = (futures[fut][0], None, futures[fut][1])
            try:
                qid, n_hits, n_doc = fut.result()
                ok += 1
            except Exception as e:  # noqa: BLE001
                err += 1
                print(f"  ERROR qid={futures[fut][0]}: {type(e).__name__}: {e}")

    dt = time.monotonic() - t0
    print(f"\nDone in {dt:.1f}s — ok={ok} err={err} ({len(queries)/dt:.1f} q/s)")


if __name__ == "__main__":
    main()
