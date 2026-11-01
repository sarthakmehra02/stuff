"""Validation script for the evaluation test set (data/test_qa.json).

Checks for:
1. Missing required fields across all rows.
2. Duplicate IDs.
3. Duplicate questions (normalized text).
4. Invalid 'type' value (must be one of VALID_TYPES).
5. answerable=False rows that have a non-empty source_pages list.
6. answerable=True rows that do NOT have a non-empty source_pages list.
7. Breakdown counts by scheme and question type.

Valid question types: direct, multi_step, cross_scheme, paraphrase, hinglish, unanswerable

Usage:
    python src/validate_test_set.py
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

# Reconfigure stdout for UTF-8 on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def find_test_qa_path() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "test_qa.json",
        PROJECT_ROOT / "test_qa.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find test_qa.json in data/ or project root.")


def validate_test_set(data_path: Path) -> bool:
    print("=" * 70)
    print("  TEST SET VALIDATION: data/test_qa.json")
    print("=" * 70)
    print(f"Loading: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    print(f"Total records loaded: {len(data)}\n")

    errors: List[str] = []
    warnings: List[str] = []

    VALID_TYPES = {
        "direct", "multi_step", "cross_scheme",
        "paraphrase", "hinglish", "unanswerable",
    }

    required_fields = ["id", "question", "gold_answer", "type", "scheme", "answerable"]

    seen_ids = set()
    seen_questions = {}

    for idx, row in enumerate(data, start=1):
        row_id = row.get("id", f"ROW_{idx}")

        # 1. Missing fields
        for field in required_fields:
            if field not in row or row[field] is None:
                errors.append(f"[{row_id}] Missing required field: {field!r}")

        # 2b. Valid type
        qtype = row.get("type")
        if qtype is not None and qtype not in VALID_TYPES:
            errors.append(
                f"[{row_id}] Invalid type {qtype!r}. "
                f"Must be one of: {sorted(VALID_TYPES)}"
            )

        # 2. Duplicate IDs
        if "id" in row and row["id"]:
            if row["id"] in seen_ids:
                errors.append(f"Duplicate ID detected: {row['id']!r}")
            seen_ids.add(row["id"])

        # 3. Duplicate questions
        if "question" in row and row["question"]:
            norm_q = " ".join(row["question"].strip().lower().split())
            if norm_q in seen_questions:
                errors.append(
                    f"Duplicate question text detected between {row_id} and {seen_questions[norm_q]}: "
                    f"{row['question'][:60]!r}"
                )
            else:
                seen_questions[norm_q] = row_id

        # 4 & 5. answerable vs source_pages logic
        is_answerable = row.get("answerable")
        source_pages = row.get("source_pages")
        source_file = row.get("source_file")

        # source_pages must be present and a list
        if "source_pages" not in row:
            errors.append(f"[{row_id}] Missing field 'source_pages' (must be a list)")
        elif not isinstance(source_pages, list):
            errors.append(
                f"[{row_id}] 'source_pages' must be a list, got {type(source_pages).__name__!r}"
            )
        else:
            # Validate each element is a positive int
            for pg in source_pages:
                if not isinstance(pg, int) or pg <= 0:
                    errors.append(
                        f"[{row_id}] source_pages contains invalid entry {pg!r} (must be positive int)"
                    )

            if is_answerable is False:
                if source_pages:  # non-empty list is wrong for unanswerable
                    errors.append(
                        f"[{row_id}] answerable=False but source_pages={source_pages!r} (must be empty list)"
                    )
                if source_file is not None:
                    warnings.append(
                        f"[{row_id}] answerable=False but has source_file={source_file!r} (expected null/None)"
                    )
            elif is_answerable is True:
                if not source_pages:
                    errors.append(
                        f"[{row_id}] answerable=True but source_pages is empty (must have at least one page)"
                    )
                if not source_file:
                    errors.append(
                        f"[{row_id}] answerable=True but source_file is missing or empty"
                    )

    # Cross-tabulation by scheme and type
    schemes = sorted(list({r.get("scheme", "UNKNOWN") for r in data}))
    types   = sorted(list({r.get("type",   "UNKNOWN") for r in data}))

    print("=" * 70)
    print("  BREAKDOWN COUNTS BY SCHEME & TYPE")
    print("=" * 70)

    col_w = 15  # wide enough for 'cross_scheme'
    header_str = f"{'Scheme':<{col_w}}" + "".join(f"{t:>{col_w}}" for t in types) + f"{'Total':>{col_w}}"
    print(header_str)
    print("-" * len(header_str))

    type_totals = {t: 0 for t in types}
    for s in schemes:
        row_str = f"{s:<{col_w}}"
        s_total = 0
        for t in types:
            cnt = sum(1 for r in data if r.get("scheme") == s and r.get("type") == t)
            type_totals[t] += cnt
            s_total += cnt
            row_str += f"{cnt:>{col_w}}"
        row_str += f"{s_total:>{col_w}}"
        print(row_str)

    print("-" * len(header_str))
    total_str = f"{'TOTAL':<{col_w}}" + "".join(f"{type_totals[t]:>{col_w}}" for t in types) + f"{len(data):>{col_w}}"
    print(total_str)

    # Answerable counts
    ans_count   = sum(1 for r in data if r.get("answerable") is True)
    unans_count = sum(1 for r in data if r.get("answerable") is False)
    print(f"\nAnswerable questions:   {ans_count}")
    print(f"Unanswerable questions: {unans_count}")
    print(f"Total questions:        {len(data)}")

    # Per-type answerable breakdown
    print()
    print(f"  {'Type':<15} {'Answerable':>12} {'Unanswerable':>14} {'Total':>7}")
    print(f"  {'-'*15} {'-'*12} {'-'*14} {'-'*7}")
    for t in types:
        t_ans   = sum(1 for r in data if r.get("type") == t and r.get("answerable") is True)
        t_unans = sum(1 for r in data if r.get("type") == t and r.get("answerable") is False)
        print(f"  {t:<15} {t_ans:>12} {t_unans:>14} {t_ans+t_unans:>7}")

    # Errors / Warnings summary
    print("\n" + "=" * 70)
    print("  VALIDATION RESULTS")
    print("=" * 70)

    if warnings:
        print(f"\n[!] {len(warnings)} Warning(s):")
        for w in warnings:
            print(f"    - {w}")

    if errors:
        print(f"\n[X] {len(errors)} Error(s) Found:")
        for e in errors:
            print(f"    - {e}")
        print("\nValidation FAILED.")
        return False
    else:
        print("\n[PASS] All validation checks passed successfully!")
        print("  - No missing fields")
        print("  - No duplicate IDs")
        print("  - No duplicate questions")
        print("  - All type values are valid")
        print("  - All answerable questions have non-empty source_pages and source_file")
        print("  - All unanswerable questions have source_pages=[]")
        return True


def main():
    target_path = find_test_qa_path()
    success = validate_test_set(target_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
