"""Merge a test-set batch file into data/test_qa.json safely.

Steps:
  1. Back up data/test_qa.json to data/backups/test_qa_<timestamp>.json.
  2. Refuse to merge if any new id OR question text already exists (case-insensitive,
     normalised whitespace).
  3. Append the batch records.
  4. Write the updated file and run validate_test_set.py.

Usage:
    python src/merge_test_batch.py <batch_file_path>

Example:
    python src/merge_test_batch.py data/test_qa_batch3.json
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
QA_PATH      = PROJECT_ROOT / "data" / "test_qa.json"
BACKUP_DIR   = PROJECT_ROOT / "data" / "backups"

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _norm_q(text: str) -> str:
    """Normalise question text for duplicate detection."""
    return " ".join(text.strip().lower().split())


def merge(batch_path: Path) -> None:
    # ── load existing ─────────────────────────────────────────────────────────
    if not QA_PATH.exists():
        print(f"ERROR: {QA_PATH} not found.")
        sys.exit(1)

    with open(QA_PATH, "r", encoding="utf-8") as f:
        existing: list = json.load(f)

    existing_ids  = {r["id"] for r in existing}
    existing_qs   = {_norm_q(r["question"]): r["id"] for r in existing}

    # ── load batch ────────────────────────────────────────────────────────────
    if not batch_path.exists():
        print(f"ERROR: batch file not found: {batch_path}")
        sys.exit(1)

    with open(batch_path, "r", encoding="utf-8") as f:
        batch: list = json.load(f)

    print(f"Batch records to merge: {len(batch)}")

    # ── conflict checks ───────────────────────────────────────────────────────
    conflicts = []
    for rec in batch:
        rid  = rec.get("id", "")
        rq   = _norm_q(rec.get("question", ""))

        if rid in existing_ids:
            conflicts.append(f"  Duplicate ID: {rid!r}")
        if rq in existing_qs:
            conflicts.append(
                f"  Duplicate question (matches {existing_qs[rq]!r}): "
                f"{rec['question'][:70]!r}"
            )

    if conflicts:
        print(f"\n[REFUSED] {len(conflicts)} conflict(s) found — merge aborted:")
        for c in conflicts:
            print(c)
        sys.exit(1)

    # ── backup ────────────────────────────────────────────────────────────────
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"test_qa_{ts}.json"
    shutil.copy2(QA_PATH, backup_path)
    print(f"Backed up to: {backup_path}")

    # ── merge and write ───────────────────────────────────────────────────────
    merged = existing + batch
    with open(QA_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(batch)} records → total now {len(merged)}.")
    print(f"Written to: {QA_PATH}\n")

    # ── run validate ──────────────────────────────────────────────────────────
    print("=" * 62)
    print("  Running validate_test_set.py …")
    print("=" * 62)
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "validate_test_set.py")],
        cwd=str(PROJECT_ROOT),
    )
    sys.exit(result.returncode)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python src/merge_test_batch.py <batch_file_path>")
        sys.exit(1)
    merge(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
