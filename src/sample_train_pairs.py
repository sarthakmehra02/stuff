"""Export a deterministic sample of synthetic pairs for human review."""

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def write_sample(input_path: Path, output_path: Path, size: int = 50, seed: int = 42):
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sample = random.Random(seed).sample(rows, min(size, len(rows)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["chunk_id", "question", "answer", "evidence_quote", "my_grade", "my_note"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sample:
            writer.writerow({key: row.get(key, "") for key in fields})
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "train_qa.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "train_sample_for_manual_check.csv")
    args = parser.parse_args()
    print(f"Wrote {len(write_sample(args.input, args.out))} rows to {args.out}")


if __name__ == "__main__":
    main()
