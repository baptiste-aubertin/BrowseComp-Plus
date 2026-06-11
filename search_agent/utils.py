import json
import random
import re
from typing import Iterable, List, Set, Tuple


def add_subsample_args(parser) -> None:
    """Register --subsample-size / --subsample-seed on an argparse parser."""
    parser.add_argument(
        "--subsample-size",
        type=int,
        default=0,
        help="Run/evaluate on a random subsample of N queries (0 = all, default).",
    )
    parser.add_argument(
        "--subsample-seed",
        type=int,
        default=42,
        help="Seed for the subsample selection (default: %(default)s).",
    )


def subsample_query_ids(qids: Iterable[str], size: int, seed: int) -> Set[str]:
    """Deterministically pick `size` query ids out of `qids`.

    Sampling happens over the sorted id set, so any two scripts given the
    same ids, size and seed (e.g. a run client and an evaluation script)
    select the same subset regardless of input ordering.
    """
    unique = sorted(set(str(q) for q in qids))
    if size <= 0 or size >= len(unique):
        return set(unique)
    return set(random.Random(seed).sample(unique, size))


def subsample_queries(
    queries: List[Tuple[str, str]], size: int, seed: int
) -> List[Tuple[str, str]]:
    """Filter a [(qid, query)] list down to a deterministic subsample, keeping order."""
    keep = subsample_query_ids((qid for qid, _ in queries), size, seed)
    return [(qid, qtext) for qid, qtext in queries if str(qid) in keep]


def extract_retrieved_docids_from_result(result: List[dict]) -> List[str]:
    retrieved_docids_set = set()
    for item in result:
        if item.get("type") != "tool_call":
            continue
        tool_name = str(item.get("tool_name") or "")
        if "search" in tool_name.lower() or "retrieval" in tool_name.lower():
            output = item.get("output")
            parsed = None
            if isinstance(output, str):
                try:
                    parsed = json.loads(output)
                except Exception:
                    parsed = None
            elif isinstance(output, list):
                parsed = output

            if isinstance(parsed, list):
                for elem in parsed:
                    if isinstance(elem, dict) and "docid" in elem:
                        retrieved_docids_set.add(str(elem["docid"]))
            else:
                # Fallback: regex grep docids from raw string output
                raw = output if isinstance(output, str) else ""
                if raw:
                    # Quoted docid values
                    for m in re.findall(r'"docid"\s*:\s*"([^"]+)"', raw):
                        retrieved_docids_set.add(str(m))
                    # Unquoted numeric docid values
                    for m in re.findall(r'"docid"\s*:\s*(\d+)', raw):
                        retrieved_docids_set.add(str(m))
    return list(retrieved_docids_set)
