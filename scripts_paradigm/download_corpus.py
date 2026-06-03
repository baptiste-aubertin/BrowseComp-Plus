"""Download Tevatron/browsecomp-plus-corpus to a local JSONL.

Output rows: {"docid": str, "url": str, "text": str}
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/browsecomp_plus_corpus.jsonl"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--repo", default="Tevatron/browsecomp-plus-corpus")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.repo, split=args.split)
    print(f"Loaded {len(ds):,} rows from {args.repo} [{args.split}]")
    print(f"Columns: {ds.column_names}")

    with args.output.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, total=len(ds), desc="writing"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
