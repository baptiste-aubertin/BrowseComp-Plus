"""Post-run analysis for a BrowseComp-Plus evaluation.

Reads an eval directory produced by scripts_evaluation/evaluate_run.py
(detailed_judge_results.csv + evaluation_summary.json) plus the run JSONs it
references, and prints:
  1. headline metrics
  2. failure decomposition (correct / wrong / no-answer, with harness sub-causes)
  3. retrieval attribution (evidence recall of correct vs wrong answers)
  4. search behavior (call counts, empty-result rate, tool errors)
  5. confidence calibration table
  6. leaderboard placement (optional, downloads official results)
  7. head-to-head against another eval dir (optional)

It also writes empty_search_queries.csv into the eval dir when any search call
returned zero results.

Usage:
  python scripts_paradigm/analyze_results.py evals/paradigm/oss-120b-skip-rerank \
      --compare evals/paradigm/oss-120b --rank
"""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def is_true(v: str) -> bool:
    return str(v).strip().lower() == "true"


def load_eval_dir(eval_dir: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((eval_dir / "evaluation_summary.json").read_text())
    with (eval_dir / "detailed_judge_results.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return summary, rows


def load_run(row: dict) -> dict:
    path = Path(row["json_path"])
    if not path.is_absolute():
        path = REPO_ROOT / path
    return json.loads(path.read_text())


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def analyze_headline(summary: dict) -> None:
    section("Headline")
    calls = summary.get("avg_tool_stats", {}).get("search")
    print(f"LLM:                {summary.get('LLM')}")
    print(f"Accuracy:           {summary.get('Accuracy (%)')}%")
    print(f"Evidence recall:    {summary.get('Recall (%)')}%")
    print(f"Search calls/query: {calls:.2f}" if calls else "Search calls/query: ?")
    print(f"Calibration error:  {summary.get('Calibration Error (%)')}%")
    print(f"Queries:            {len(summary.get('per_query_metrics', []))}")


def analyze_failures(rows: list[dict], runs: dict[str, dict]) -> None:
    section("Failure decomposition")
    n = len(rows)
    correct = [r for r in rows if is_true(r["judge_correct"])]
    answered = [r for r in rows if r["predicted_answer"].strip()]
    wrong = [r for r in answered if not is_true(r["judge_correct"])]
    no_answer = [r for r in rows if not r["predicted_answer"].strip()]
    incomplete = [r for r in no_answer if not is_true(r["is_completed"])]
    empty_completed = [r for r in no_answer if is_true(r["is_completed"])]

    print(f"correct:            {len(correct)} ({100 * len(correct) / n:.1f}%)")
    print(f"answered but wrong: {len(wrong)} ({100 * len(wrong) / n:.1f}%)")
    print(f"no answer:          {len(no_answer)} ({100 * len(no_answer) / n:.1f}%)")
    print(f"  - hit iteration limit (status=incomplete): {len(incomplete)}")
    print(f"  - 'completed' with empty final message:    {len(empty_completed)}")
    if answered:
        acc = 100 * len(correct) / len(answered)
        print(f"answered-only accuracy: {acc:.1f}%  "
              f"(harness ceiling: +{acc - 100 * len(correct) / n:.1f} pts)")

    if no_answer:
        searches = [sum(runs[r["query_id"]]["tool_call_counts"].values()) for r in no_answer]
        print(f"no-answer runs searched median {statistics.median(searches):.0f} times "
              f"(mean {statistics.mean(searches):.1f})")


def analyze_retrieval_attribution(summary: dict) -> None:
    section("Retrieval attribution (evidence recall vs correctness)")
    pq = summary["per_query_metrics"]
    right = [q for q in pq if q["correct"]]
    wrong = [q for q in pq if not q["correct"]]
    if not right or not wrong:
        print("not enough data")
        return
    print(f"correct answers: avg recall {statistics.mean(q['recall'] for q in right):.1f}%")
    print(f"wrong answers:   avg recall {statistics.mean(q['recall'] for q in wrong):.1f}%")
    zero = sum(1 for q in wrong if q["recall"] == 0)
    full = sum(1 for q in wrong if q["recall"] == 100)
    print(f"wrong with recall=0 (pure retrieval failure):  {zero} ({100 * zero / len(wrong):.0f}%)")
    print(f"wrong with recall=100 (pure reasoning failure): {full} ({100 * full / len(wrong):.0f}%)")


def analyze_search_behavior(rows: list[dict], runs: dict[str, dict], eval_dir: Path) -> None:
    section("Search behavior")

    groups = defaultdict(list)
    for r in rows:
        if is_true(r["judge_correct"]):
            g = "correct"
        elif r["predicted_answer"].strip():
            g = "wrong"
        else:
            g = "no_answer"
        groups[g].append(sum(runs[r["query_id"]]["tool_call_counts"].values()))
    for g in ("correct", "wrong", "no_answer"):
        v = groups.get(g)
        if v:
            print(f"searches/run [{g:>9}]: mean {statistics.mean(v):5.1f}  "
                  f"median {statistics.median(v):4.0f}  max {max(v)}")

    total = empty = errors = 0
    empty_rows = []
    error_samples = Counter()
    for run in runs.values():
        for i, item in enumerate(run["result"]):
            if item["type"] != "tool_call":
                continue
            total += 1
            out = item["output"]
            if out is None or str(out).strip() in ("[]", ""):
                empty += 1
                try:
                    q = json.loads(item["arguments"]).get("user_query", "")
                except Exception:
                    q = item["arguments"]
                empty_rows.append({"query_id": run.get("query_id"), "search_query": q,
                                   "tool_call_index": i})
            elif isinstance(out, str) and out.startswith("Error"):
                errors += 1
                error_samples[out[:100]] += 1

    print(f"\nsearch calls: {total} | empty results: {empty} ({100 * empty / total:.1f}%)"
          f" | tool errors: {errors}")
    for msg, c in error_samples.most_common(3):
        print(f"  {c}x {msg}")

    if empty_rows:
        out_path = eval_dir / "empty_search_queries.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["query_id", "search_query", "tool_call_index"])
            w.writeheader()
            w.writerows(empty_rows)
        print(f"  -> wrote {len(empty_rows)} empty-result queries to {out_path}")


def analyze_calibration(rows: list[dict]) -> None:
    section("Calibration (confidence vs accuracy)")
    buckets = defaultdict(lambda: [0, 0])
    for r in rows:
        if not r["confidence"].strip():
            continue
        k = min(int(float(r["confidence"]) // 10) * 10, 90)
        buckets[k][0] += 1
        buckets[k][1] += is_true(r["judge_correct"])
    print(f"{'confidence':>10} | {'n':>4} | accuracy")
    for k in sorted(buckets):
        n, ok = buckets[k]
        print(f"{f'{k}-{k + 9}':>10} | {n:>4} | {100 * ok / n:.0f}%")


def analyze_head_to_head(summary: dict, eval_dir: Path, compare_dir: Path) -> None:
    section(f"Head-to-head vs {compare_dir.name}")
    other = json.loads((compare_dir / "evaluation_summary.json").read_text())
    ours = {q["query_id"]: q for q in summary["per_query_metrics"]}
    theirs = {q["query_id"]: q for q in other["per_query_metrics"]}
    common = sorted(set(ours) & set(theirs))
    if not common:
        print("no common queries")
        return
    both = sum(1 for q in common if ours[q]["correct"] and theirs[q]["correct"])
    only_ours = sum(1 for q in common if ours[q]["correct"] and not theirs[q]["correct"])
    only_theirs = sum(1 for q in common if theirs[q]["correct"] and not ours[q]["correct"])
    print(f"common queries: {len(common)}")
    print(f"both correct: {both} | only {eval_dir.name}: {only_ours} | "
          f"only {compare_dir.name}: {only_theirs} | neither: {len(common) - both - only_ours - only_theirs}")
    delta = statistics.mean(ours[q]["recall"] - theirs[q]["recall"] for q in common)
    print(f"mean per-query recall delta ({eval_dir.name} - {compare_dir.name}): {delta:+.1f}")


def analyze_rank(eval_dir: Path) -> None:
    section("Leaderboard placement")
    sys.path.insert(0, str(REPO_ROOT / "scripts_paradigm"))
    try:
        import leaderboard_rank as lb

        board = lb.load_leaderboard(None)
        ours = lb.load_ours(str(eval_dir / "evaluation_summary.json"),
                            "oss-120b-high", f"LightOn Paradigm ({eval_dir.name})")
        overall = board + [ours]
        standard = [r for r in overall if not r["scaffold"]]
        same_llm = [r for r in standard if r["LLM"] == "oss-120b-high" or r.get("ours")]
        print(f"overall (all scaffolds):      {lb.rank_of(ours, overall)} / {len(overall)}")
        print(f"standard scaffold:            {lb.rank_of(ours, standard)} / {len(standard)}")
        print(f"oss-120b-high retrievers:     {lb.rank_of(ours, same_llm)} / {len(same_llm)}")
        lb.print_table(same_llm, "Retriever comparison (oss-120b-high, Standard):")
    except Exception as e:
        print(f"could not compute leaderboard rank: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", type=Path,
                        help="Eval directory (with detailed_judge_results.csv + evaluation_summary.json)")
    parser.add_argument("--compare", type=Path, default=None,
                        help="Another eval directory for a per-query head-to-head")
    parser.add_argument("--rank", action="store_true",
                        help="Also compute leaderboard placement (downloads official results)")
    args = parser.parse_args()

    summary, rows = load_eval_dir(args.eval_dir)
    runs = {r["query_id"]: load_run(r) for r in rows}

    print(f"Analyzing {args.eval_dir} ({len(rows)} queries)")
    analyze_headline(summary)
    analyze_failures(rows, runs)
    analyze_retrieval_attribution(summary)
    analyze_search_behavior(rows, runs, args.eval_dir)
    analyze_calibration(rows)
    if args.compare:
        analyze_head_to_head(summary, args.eval_dir, args.compare)
    if args.rank:
        analyze_rank(args.eval_dir)


if __name__ == "__main__":
    main()
