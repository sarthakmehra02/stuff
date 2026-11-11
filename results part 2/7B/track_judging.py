"""Print progress and current verdict counts for the 7B blind judge run."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in sorted(ROOT.glob("judged_*.jsonl")):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = {}
    for row in rows:
        label = row.get("verdict", "missing")
        counts[label] = counts.get(label, 0) + 1
    print(f"{path.name}: {len(rows)}/60 | {counts}")
