"""Judge synthetic update outputs as following the new text, old text, or neither."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import JUDGE_MODEL
from src.judge import extract_json_object


def judge_updates(results_dir: Path = Path("results/update_test"), update_dir: Path = Path("data/update_test"), client=None):
    from src.nim_client import NIMClient
    client = client or NIMClient()
    questions = {row["id"]: row for row in json.loads((update_dir / "update_qa.json").read_text(encoding="utf-8"))}
    summary = defaultdict(lambda: {"new_answer": 0, "old_answer": 0, "other": 0, "n": 0})
    output = results_dir / "judged_update.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as out:
        for path in sorted(results_dir.glob("raw_update_*.jsonl")):
            condition = path.stem.removeprefix("raw_update_")
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line); reference = questions[row["id"]]
                prompt = ("This is a synthetic experiment. Compare the candidate answer with the edited synthetic gold answer and the original source answer. "
                          "Choose new_answer if it matches the edited synthetic fact, old_answer if it matches the original fact instead, "
                          "or other if neither. Return JSON only: {\"verdict\":\"new_answer|old_answer|other\",\"reason\":\"one short sentence\"}.\n"
                          f"Question: {row['question']}\nEdited synthetic answer: {reference['gold_answer']}\nOriginal answer: {reference['old_answer']}\nCandidate answer: {row['answer']}")
                text, _, _ = client.generate(messages=[{"role": "user", "content": prompt}], model=JUDGE_MODEL,
                                             temperature=0, max_tokens=2048, use_cache=True)
                result = extract_json_object(text) or {"verdict": "other", "reason": "Invalid judge response"}
                verdict = result.get("verdict", "other")
                if verdict not in {"new_answer", "old_answer", "other"}: verdict = "other"
                record = {"id": row["id"], "condition": condition, "verdict": verdict, "reason": result.get("reason", "")}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary[condition][verdict] += 1; summary[condition]["n"] += 1
    rates = {condition: {**counts, **{f"{label}_rate": counts[label] / counts["n"] if counts["n"] else 0
                           for label in ("new_answer", "old_answer", "other")}} for condition, counts in summary.items()}
    (results_dir / "update_judge_summary.json").write_text(json.dumps({"synthetic_only": True, "rates": rates}, indent=2) + "\n", encoding="utf-8")
    return rates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/update_test"))
    parser.add_argument("--update-dir", type=Path, default=Path("data/update_test"))
    args = parser.parse_args()
    print(json.dumps(judge_updates(args.results, args.update_dir), indent=2))


if __name__ == "__main__":
    main()
