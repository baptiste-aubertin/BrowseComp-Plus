"""Compute our placement on the BrowseComp-Plus leaderboard.

Downloads the official leaderboard results CSV (the same file the HF Space
Tevatron/BrowseComp-Plus renders), inserts our run from evaluation_summary.json,
and prints:
  1. overall rank (all scaffolds)
  2. rank among Standard-scaffold entries (search-only, top-5)
  3. retriever table for the same LLM on the Standard scaffold

Usage:
  python scripts_paradigm/leaderboard_rank.py \
      --summary evals/paradigm/oss-120b/evaluation_summary.json \
      --llm-name oss-120b-high --retriever-name "LightOn Paradigm"
"""

import argparse
import csv
import io
import json
import urllib.request

RESULTS_CSV_URL = (
    "https://huggingface.co/datasets/Tevatron/BrowseComp-Plus-results/"
    "resolve/main/agent_results.csv"
)


def load_leaderboard(csv_path: str | None) -> list[dict]:
    if csv_path:
        with open(csv_path, encoding="utf-8-sig") as f:
            text = f.read()
    else:
        with urllib.request.urlopen(RESULTS_CSV_URL) as r:
            text = r.read().decode("utf-8-sig")

    rows = [r for r in csv.DictReader(io.StringIO(text)) if r["Accuracy (%)"].strip()]
    for r in rows:
        r["acc"] = float(r["Accuracy (%)"])
        r["scaffold"] = (r.get("Scaffold") or "").strip()

    # The CSV contains exact duplicate entries (e.g. oss-120b-high + BM25 appears
    # twice with identical numbers); keep one of each so ranks aren't inflated.
    seen, deduped = set(), []
    for r in rows:
        key = (r["LLM"], r["Retriever"], r["acc"], r["scaffold"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def load_ours(summary_path: str, llm_name: str, retriever_name: str) -> dict:
    with open(summary_path, encoding="utf-8") as f:
        s = json.load(f)
    return {
        "LLM": llm_name,
        "Retriever": retriever_name,
        "acc": float(s["Accuracy (%)"]),
        "Recall (%)": str(s["Recall (%)"]),
        "Search Calls": f"{s['avg_tool_stats']['search']:.2f}",
        "scaffold": "",  # search-only top-5 == Standard
        "ours": True,
    }


def rank_of(entry: dict, pool: list[dict]) -> int:
    return sum(1 for r in pool if r["acc"] > entry["acc"]) + 1


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"{'rank':>4}  {'acc':>6}  {'recall':>6}  {'calls':>6}  {'LLM':<30} Retriever")
    for i, r in enumerate(sorted(rows, key=lambda r: -r["acc"]), 1):
        marker = " <-- ours" if r.get("ours") else ""
        print(
            f"{i:>4}  {r['acc']:>6.2f}  {r['Recall (%)']:>6}  {r['Search Calls']:>6}"
            f"  {r['LLM']:<30} {r['Retriever']}{marker}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="evals/paradigm/oss-120b/evaluation_summary.json")
    parser.add_argument("--csv", default=None, help="Local agent_results.csv (skips download)")
    parser.add_argument("--llm-name", default="oss-120b-high",
                        help="Our LLM as named on the leaderboard, for the same-LLM table")
    parser.add_argument("--retriever-name", default="LightOn Paradigm")
    args = parser.parse_args()

    board = load_leaderboard(args.csv)
    ours = load_ours(args.summary, args.llm_name, args.retriever_name)

    overall = board + [ours]
    standard = [r for r in overall if not r["scaffold"]]
    same_llm_standard = [r for r in standard if r["LLM"] == args.llm_name or r.get("ours")]

    print(f"Our entry: {ours['LLM']} + {ours['Retriever']} | "
          f"accuracy {ours['acc']} | recall {ours['Recall (%)']} | "
          f"search calls {ours['Search Calls']}")
    print(f"\nOverall rank (all scaffolds):     {rank_of(ours, overall)} / {len(overall)}")
    print(f"Rank among Standard scaffold:     {rank_of(ours, standard)} / {len(standard)}")
    print(f"Rank among {args.llm_name} (Standard): "
          f"{rank_of(ours, same_llm_standard)} / {len(same_llm_standard)}")

    print_table(same_llm_standard, f"Retriever comparison — {args.llm_name}, Standard scaffold:")
    print_table(standard, "Full Standard-scaffold ranking:")


if __name__ == "__main__":
    main()
