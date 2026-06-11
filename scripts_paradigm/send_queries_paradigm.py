#!/usr/bin/env python3
"""Send a file of search queries to a LightOn Paradigm instance, concurrently.

Standalone & shareable: only dependency is `requests` (python-dotenv is used
if installed, to pick up a local .env). Reads queries from a TSV/text file
(one query per line; if a line contains tabs, the LAST column is used as the
query text and the first as its id) and POSTs each one to /api/v3/search.

Usage:
    python send_queries_paradigm.py queries.tsv \
        --base-url https://my-instance.lighton.ai --api-key sk-... \
        --workers 20 --limit 500 --output results.jsonl

Credentials default to $PARADIGM_BASE_URL / $PARADIGM_API_KEY /
$PARADIGM_WORKSPACE_ID.
"""

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

try:  # optional: load a .env next to the script or in cwd
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    load_dotenv(override=False)
except ImportError:
    pass

logger = logging.getLogger("send_queries_paradigm")

BACKOFF_BASE = 1.0


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("query_file", help="TSV/text file with one query per line.")

    conn = p.add_argument_group("connection")
    conn.add_argument(
        "--base-url",
        default=os.getenv("PARADIGM_BASE_URL"),
        help="Paradigm base URL (default: $PARADIGM_BASE_URL).",
    )
    conn.add_argument(
        "--api-key",
        default=os.getenv("PARADIGM_API_KEY"),
        help="Paradigm API key (default: $PARADIGM_API_KEY).",
    )
    conn.add_argument(
        "--workspace-id",
        type=int,
        default=int(os.environ["PARADIGM_WORKSPACE_ID"])
        if os.getenv("PARADIGM_WORKSPACE_ID")
        else None,
        help="Restrict search to this workspace (default: $PARADIGM_WORKSPACE_ID).",
    )
    conn.add_argument(
        "--timeout", type=float, default=120.0, help="Per-request HTTP timeout in seconds."
    )

    load = p.add_argument_group("load shape")
    load.add_argument("--workers", type=int, default=10, help="Concurrent threads (default: 10).")
    load.add_argument(
        "--limit", type=int, default=0, help="Number of queries to send (0 = all, default)."
    )
    load.add_argument("--offset", type=int, default=0, help="Skip the first N queries.")
    load.add_argument(
        "--shuffle", action="store_true", help="Shuffle queries before applying offset/limit."
    )
    load.add_argument("--seed", type=int, default=0, help="Shuffle seed (default: 0).")
    load.add_argument(
        "--qps",
        type=float,
        default=0.0,
        help="Global cap on query submission rate, queries/sec (0 = unlimited).",
    )

    search = p.add_argument_group("search parameters")
    search.add_argument(
        "--max-results", type=int, default=50, help="max_results per search (default: 50)."
    )
    search.add_argument(
        "--mode", choices=["text", "vision"], default="text", help="Search mode (default: text)."
    )
    search.add_argument(
        "--relevance-scoring",
        choices=["scoring_and_filtering", "scoring_only", "none"],
        default="scoring_and_filtering",
        help=(
            "Cross-encoder mode: 'scoring_and_filtering' (default; threshold-filtered "
            "with adaptive fallback on empty), 'scoring_only' (score all, no filtering), "
            "'none' (skip scoring; fastest)."
        ),
    )

    out = p.add_argument_group("output")
    out.add_argument(
        "--output",
        help="Write one JSON line per query ({qid, query, status, latency_s, n_results, results}).",
    )
    out.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Retries per query on 429/5xx/connection errors before counting it as failed "
        "(-1 = retry forever, default: 10).",
    )
    out.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p.parse_args()


def load_queries(path, offset, limit, shuffle, seed):
    """Return [(qid, query)] — qid is the first TSV column if present, else the line number."""
    queries = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" in line:
                qid, _, text = line.partition("\t")
                queries.append((qid, text.split("\t")[-1]))
            else:
                queries.append((str(i), line))
    if shuffle:
        random.Random(seed).shuffle(queries)
    queries = queries[offset:]
    if limit > 0:
        queries = queries[:limit]
    return queries


class RateLimiter:
    """Simple global pacing: blocks so submissions average at most `qps` per second."""

    def __init__(self, qps):
        self.interval = 1.0 / qps if qps > 0 else 0.0
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def wait(self):
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next - now
            self._next = max(self._next, now) + self.interval
        if sleep_for > 0:
            time.sleep(sleep_for)


def search_once(session, args, query):
    """POST one query; retries transient failures. Returns (status_code, response_json|None, retries)."""
    payload = {
        "query": query,
        "max_results": args.max_results,
        "mode": args.mode,
        "relevance_scoring": args.relevance_scoring,
    }
    if args.workspace_id is not None:
        payload["workspace_id"] = [args.workspace_id]

    attempt = 0
    while True:
        try:
            r = session.post(
                f"{args.base_url}/api/v3/search", json=payload, timeout=args.timeout
            )
        except requests.RequestException as exc:
            if args.max_retries >= 0 and attempt >= args.max_retries:
                raise
            wait = BACKOFF_BASE * (2 ** attempt)
            attempt += 1
            logger.warning("connection error (%s); retry %d in %.1fs", exc, attempt, wait)
            time.sleep(wait)
            continue

        if r.ok:
            return r.status_code, r.json(), attempt

        if r.status_code == 429 or r.status_code >= 500:
            if args.max_retries >= 0 and attempt >= args.max_retries:
                return r.status_code, None, attempt
            retry_after = r.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else BACKOFF_BASE * (2 ** attempt)
            except ValueError:
                wait = BACKOFF_BASE * (2 ** attempt)
            attempt += 1
            logger.warning("HTTP %d; retry %d in %.1fs", r.status_code, attempt, wait)
            time.sleep(wait)
            continue

        # Other 4xx are deterministic — don't retry.
        logger.error("HTTP %d for query %r: %s", r.status_code, query[:80], r.text[:300])
        return r.status_code, None, attempt


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.base_url:
        sys.exit("error: --base-url is required (or set $PARADIGM_BASE_URL)")
    if not args.api_key:
        sys.exit("error: --api-key is required (or set $PARADIGM_API_KEY)")
    args.base_url = args.base_url.rstrip("/")

    queries = load_queries(args.query_file, args.offset, args.limit, args.shuffle, args.seed)
    if not queries:
        sys.exit("error: no queries to send")
    print(
        f"Sending {len(queries)} queries to {args.base_url} "
        f"(workers={args.workers}, workspace={args.workspace_id}, mode={args.mode}, "
        f"max_results={args.max_results}, relevance_scoring={args.relevance_scoring}, "
        f"qps={args.qps or 'unlimited'})"
    )

    session = requests.Session()
    session.headers["X-Api-Key"] = args.api_key
    # requests pools 10 connections per host by default; match the worker count.
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=args.workers, pool_maxsize=args.workers
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    limiter = RateLimiter(args.qps)
    out_file = open(args.output, "w", encoding="utf-8") if args.output else None
    out_lock = threading.Lock()

    ok = failed = done = 0
    total_retries = 0
    latencies = []
    stats_lock = threading.Lock()
    t0 = time.monotonic()

    def run_one(item):
        qid, query = item
        limiter.wait()
        t = time.monotonic()
        status, body, retries = search_once(session, args, query)
        latency = time.monotonic() - t
        results = (body or {}).get("results", [])
        if out_file:
            record = {
                "qid": qid,
                "query": query,
                "status": status,
                "latency_s": round(latency, 3),
                "n_results": len(results),
                "results": results,
            }
            with out_lock:
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return status, body is not None, latency, retries

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(run_one, q): q for q in queries}
            for fut in as_completed(futures):
                qid, query = futures[fut]
                try:
                    status, success, latency, retries = fut.result()
                except Exception as e:  # noqa: BLE001
                    success, latency, retries = False, 0.0, 0
                    logger.error("qid=%s failed: %s: %s", qid, type(e).__name__, e)
                with stats_lock:
                    done += 1
                    total_retries += retries
                    if success:
                        ok += 1
                        latencies.append(latency)
                    else:
                        failed += 1
                    if done % 50 == 0 or done == len(queries):
                        dt = time.monotonic() - t0
                        print(
                            f"  {done}/{len(queries)} done — ok={ok} failed={failed} "
                            f"retries={total_retries} ({done / dt:.1f} q/s)"
                        )
    finally:
        if out_file:
            out_file.close()

    dt = time.monotonic() - t0
    print(f"\nDone in {dt:.1f}s — ok={ok} failed={failed} retries={total_retries} ({len(queries) / dt:.2f} q/s)")
    if latencies:
        latencies.sort()
        pct = lambda q: latencies[min(len(latencies) - 1, int(q * len(latencies)))]  # noqa: E731
        print(
            f"latency: mean={sum(latencies) / len(latencies):.2f}s "
            f"p50={pct(0.50):.2f}s p90={pct(0.90):.2f}s p99={pct(0.99):.2f}s max={latencies[-1]:.2f}s"
        )
    if args.output:
        print(f"results written to {args.output}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
