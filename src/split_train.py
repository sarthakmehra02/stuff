"""Split synthetic examples by source chunk to prevent within-chunk leakage."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def split_by_chunk(input_path: Path, output_dir: Path, seed: int = 42, val_fraction: float = 0.05):
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["chunk_id"]].append(row)
    chunks = sorted(grouped)
    random.Random(seed).shuffle(chunks)
    val_count = max(1, round(len(chunks) * val_fraction)) if len(chunks) > 1 else 0
    val_chunks = set(chunks[:val_count])
    train = [row for chunk in chunks if chunk not in val_chunks for row in grouped[chunk]]
    val = [row for chunk in chunks if chunk in val_chunks for row in grouped[chunk]]
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("train.jsonl", train), ("val.jsonl", val)):
        with (output_dir / name).open("w", encoding="utf-8", newline="\n") as stream:
            for row in data:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    assert not ({row["chunk_id"] for row in train} & {row["chunk_id"] for row in val})
    return train, val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "train_qa.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "train_split")
    args = parser.parse_args()
    train, val = split_by_chunk(args.input, args.out_dir)
    print(f"train={len(train)} val={len(val)}; chunks are disjoint")


if __name__ == "__main__":
    main()
