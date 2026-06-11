"""Re-run a query from an empty_search_queries.csv against ParadigmSearcher.

Example:
    python scripts_paradigm/run_empty_query.py --query-id 770
    python scripts_paradigm/run_empty_query.py --query-id 770 --skip-rerank
"""

import argparse
import csv
import importlib.util
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

DEFAULT_CSV = ROOT / "evals" / "paradigm" / "oss-120b" / "empty_search_queries.csv"


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
    parser = argparse.ArgumentParser(description=__doc__)
    ParadigmSearcher.parse_args(parser)  # adds --base-url, --api-key, --skip-rerank, etc.
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"CSV file (default: {DEFAULT_CSV})")
    parser.add_argument("--query-id", required=True, help="query_id to run (all CSV rows with this id).")
    parser.add_argument("--row", type=int, default=None,
                        help="When the query_id has several rows, run only this one (0-based).")
    parser.add_argument("--k", type=int, default=5, help="Number of results to show (default: 5).")
    parser.add_argument("--strip-quotes", action="store_true",
                        help='Remove " characters from the query before searching.')
    args = parser.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["query_id"] == str(args.query_id)]

    if not rows:
        sys.exit(f"No rows with query_id={args.query_id} in {args.csv}")
    if args.row is not None:
        if not 0 <= args.row < len(rows):
            sys.exit(f"--row {args.row} out of range: query_id={args.query_id} has {len(rows)} row(s)")
        rows = [rows[args.row]]

    searcher = ParadigmSearcher(args)
    print(f"query_id={args.query_id}: {len(rows)} quer(y/ies), skip_rerank={args.skip_rerank}")

    for n, row in enumerate(rows):
        query = row["search_query"]
        if args.strip_quotes:
            query = query.replace('"', "")
        print(f"\n=== [{n}] search(query={query!r}, k={args.k}) ===")
        print(f"    (from {row['run_file']} tool_call_index={row['tool_call_index']})")
        hits = searcher.search(query, k=args.k)
        if not hits:
            print("  -> EMPTY (0 hits)")
            continue
        for i, hit in enumerate(hits, 1):
            snippet = (hit.get("text") or "").replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "…"
            print(f"  [{i}] docid={hit['docid']}  score={hit['score']:.4f}")
            print(f"      {snippet}")


if __name__ == "__main__":
    main()
