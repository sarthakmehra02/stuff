"""One-off migration: source_page (int|null) -> source_pages (list[int]).

- Backs up data/test_qa.json to data/test_qa.json.bak
- For each record:
    answerable=True  -> source_pages = [source_page]  (single-element list)
    answerable=False -> source_pages = []              (empty list)
    null source_page -> source_pages = []
- Removes the old source_page field.

Usage:
    python src/migrate_source_page.py
"""

import json
import shutil
import sys
from pathlib import Path

# Reconfigure stdout for UTF-8 on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "test_qa.json"
BACKUP_PATH = PROJECT_ROOT / "data" / "test_qa.json.bak"


def migrate():
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found.")
        sys.exit(1)

    # 1. Backup
    shutil.copy2(DATA_PATH, BACKUP_PATH)
    print(f"Backed up to: {BACKUP_PATH}")

    # 2. Load
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records.")

    # 3. Migrate
    converted = 0
    for rec in records:
        old_page = rec.pop("source_page", None)   # remove old field

        if old_page is not None and isinstance(old_page, int):
            rec["source_pages"] = [old_page]
        else:
            rec["source_pages"] = []
        converted += 1

    print(f"Converted {converted} records:  source_page -> source_pages")

    # Sanity counts
    non_empty = sum(1 for r in records if r["source_pages"])
    empty = sum(1 for r in records if not r["source_pages"])
    print(f"  Non-empty source_pages: {non_empty}")
    print(f"  Empty source_pages:     {empty}")

    # 4. Write back (pretty-printed, 2-space indent)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Written back to: {DATA_PATH}")
    print("Migration complete.")


if __name__ == "__main__":
    migrate()
