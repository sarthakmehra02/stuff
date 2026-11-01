"""Create and score a stratified human review sample for the blind judge."""

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["id", "condition", "question", "gold_answer", "model_answer", "judge_verdict", "my_verdict"]


def make_sample(results_dir: Path = ROOT / "results", out_path: Path | None = None, size: int = 20, seed: int = 42):
    candidates = []
    for file in sorted(results_dir.glob("judged_*.jsonl")):
        condition = file.stem.removeprefix("judged_")
        raw_path = results_dir / f"raw_{condition}.jsonl"
        raw = {row["id"]: row for row in (json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip())}
        for verdict in (json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()):
            row = raw.get(verdict["id"])
            if row:
                candidates.append({"id": row["id"], "condition": condition, "question": row["question"],
                                   "gold_answer": row["gold_answer"], "model_answer": row["answer"],
                                   "judge_verdict": verdict["verdict"], "my_verdict": ""})
    if not candidates:
        raise ValueError("No matched raw and judged rows found")
    rng = random.Random(seed)
    picked = []
    picked_keys = set()
    test_rows = {row["id"]: row for row in json.loads((ROOT / "data" / "test_qa.json").read_text(encoding="utf-8"))}
    for category in ("unanswerable", "hinglish"):
        matches = [row for row in candidates if test_rows[row["id"]].get("type") == category]
        rng.shuffle(matches)
        for row in matches[:3]:
            key = (row["id"], row["condition"])
            if key not in picked_keys:
                picked.append(row); picked_keys.add(key)
    remaining = [row for row in candidates if (row["id"], row["condition"]) not in picked_keys]
    rng.shuffle(remaining)
    picked.extend(remaining[:max(0, size - len(picked))])
    picked = picked[:size]
    out_path = out_path or results_dir / "manual_check_sample.csv"
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing review file: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(picked)
    return picked


def score_sample(path: Path, out_path: Path | None = None):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("my_verdict", "").strip()]
    if not rows:
        raise ValueError("Fill at least one my_verdict before scoring")
    actual = [row["judge_verdict"].strip().lower() for row in rows]
    human = [row["my_verdict"].strip().lower() for row in rows]
    labels = sorted(set(actual) | set(human))
    observed = sum(a == b for a, b in zip(actual, human)) / len(rows)
    expected = sum(actual.count(label) * human.count(label) for label in labels) / (len(rows) ** 2)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    result = {"n_scored": len(rows), "percent_agreement": observed * 100, "cohen_kappa": kappa}
    target = out_path or path.parent / "judge_agreement.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "manual_check_sample.csv")
    args = parser.parse_args()
    print(json.dumps(score_sample(args.input) if args.score else {"sampled": len(make_sample())}, indent=2))


if __name__ == "__main__":
    main()
