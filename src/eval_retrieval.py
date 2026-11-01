"""Retrieval Evaluation Script for RAG on Indian Farmer Scheme Guidelines.

Evaluates retrieval quality over answerable questions from data/test_qa.json:
- Evaluates top-4 retrieval via retrieve(question, k=4).
- A retrieval is counted as a 'HIT' if any chunk matches:
    retrieved_chunk['source_file'] == expected_file AND
    abs(retrieved_chunk['page'] - p) <= 1 for ANY p in source_pages
- Computes Hit@1 and Hit@4 overall, by scheme, and by question type.
- Prints details for every missed question (ID, Question, Expected, Retrieved).
- Saves full per-query evaluation results to results/retrieval_eval.csv.

Supported question types: direct, multi_step, cross_scheme, paraphrase, hinglish
(unanswerable questions are skipped — they have no source_pages to evaluate against)

Usage:
    python src/eval_retrieval.py
"""

import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Reconfigure stdout/stderr for UTF-8 on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import retrieve, load_index


def find_test_qa_path() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "test_qa.json",
        PROJECT_ROOT / "test_qa.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find test_qa.json in data/ or project root.")


def check_hit(
    chunk: Dict[str, Any],
    expected_file: str,
    source_pages: List[int],
) -> bool:
    """Return True if chunk matches source_file and its page is within +/- 1
    of ANY page in source_pages."""
    chunk_file = str(chunk.get("source_file", "")).strip()
    chunk_page = chunk.get("page")
    if chunk_page is None or not source_pages:
        return False
    try:
        page_num = int(chunk_page)
    except (ValueError, TypeError):
        return False

    same_file = (chunk_file.lower() == expected_file.strip().lower())
    page_near = any(abs(page_num - p) <= 1 for p in source_pages)
    return same_file and page_near


def run_retrieval_evaluation():
    print("=" * 80)
    print("  RETRIEVAL EVALUATION (Hit@1 & Hit@4)")
    print("=" * 80)

    # 1. Load test dataset
    data_path = find_test_qa_path()
    with open(data_path, "r", encoding="utf-8") as f:
        all_qa: List[Dict[str, Any]] = json.load(f)

    answerable_qa = [q for q in all_qa if q.get("answerable") is True]
    print(f"Loaded {len(all_qa)} total questions from {data_path.name}")
    print(f"Evaluating {len(answerable_qa)} answerable questions (k=4, page tolerance +/- 1)\n")

    # 2. Pre-load FAISS index once for fast evaluation
    index, chunks = load_index()
    print(f"FAISS index loaded ({index.ntotal} vectors, {len(chunks)} chunks)\n")

    # 3. Evaluate each answerable query
    eval_rows: List[Dict[str, Any]] = []

    for i, item in enumerate(answerable_qa, start=1):
        qid = item["id"]
        question = item["question"]
        scheme = item.get("scheme", "UNKNOWN")
        qtype = item.get("type", "UNKNOWN")
        expected_file = str(item.get("source_file", "")).strip()
        source_pages: List[int] = item.get("source_pages") or []

        # Retrieve top 4 chunks
        retrieved_chunks = retrieve(question, k=4, index=index, chunks=chunks)

        # Check Hit@1
        hit_at_1 = False
        hit_rank = None
        if retrieved_chunks and check_hit(retrieved_chunks[0], expected_file, source_pages):
            hit_at_1 = True
            hit_rank = 1

        # Check Hit@4 and locate first hit rank
        hit_at_4 = False
        for rank, c in enumerate(retrieved_chunks, start=1):
            if check_hit(c, expected_file, source_pages):
                hit_at_4 = True
                if hit_rank is None:
                    hit_rank = rank
                break

        # Summary strings of retrieved chunks
        retrieved_summary_list = []
        retrieved_ids = []
        retrieved_scores = []
        retrieved_pages_simple = []

        for c in retrieved_chunks:
            c_file = c.get("source_file", "unknown")
            c_page = c.get("page", "?")
            c_score = c.get("score", 0.0)
            c_id = c.get("id", "")
            retrieved_summary_list.append(f"{c_file}:p{c_page} (score={c_score:.4f})")
            retrieved_pages_simple.append(f"{c_file}:p{c_page}")
            retrieved_ids.append(str(c_id))
            retrieved_scores.append(f"{c_score:.4f}")

        eval_rows.append({
            "id": qid,
            "question": question,
            "scheme": scheme,
            "type": qtype,
            "expected_file": expected_file,
            "expected_pages": ", ".join(str(p) for p in source_pages),
            "hit_at_1": 1 if hit_at_1 else 0,
            "hit_at_4": 1 if hit_at_4 else 0,
            "hit_rank": hit_rank if hit_rank is not None else "",
            "retrieved_pages_summary": "; ".join(retrieved_summary_list),
            "retrieved_pages": ", ".join(retrieved_pages_simple),
            "retrieved_chunk_ids": ", ".join(retrieved_ids),
            "retrieved_scores": ", ".join(retrieved_scores),
        })

    total_n = len(eval_rows)
    overall_h1 = sum(r["hit_at_1"] for r in eval_rows)
    overall_h4 = sum(r["hit_at_4"] for r in eval_rows)

    # 4. Print Overall Metrics
    print("=" * 80)
    print("  OVERALL RETRIEVAL PERFORMANCE")
    print("=" * 80)
    print(f"Total Evaluated Questions: {total_n}")
    print(f"Hit@1: {overall_h1:2d} / {total_n:2d} ({overall_h1 / total_n * 100:6.2f}%)")
    print(f"Hit@4: {overall_h4:2d} / {total_n:2d} ({overall_h4 / total_n * 100:6.2f}%)")
    print("=" * 80)

    # 5. Breakdown by Scheme
    print("\n" + "=" * 80)
    print("  PERFORMANCE BY SCHEME")
    print("=" * 80)
    schemes = sorted(list({r["scheme"] for r in eval_rows}))
    print(f"{'Scheme':<15} {'Total':>8} {'Hit@1':>8} {'Hit@1 (%)':>12} {'Hit@4':>8} {'Hit@4 (%)':>12}")
    print("-" * 65)
    for s in schemes:
        s_rows = [r for r in eval_rows if r["scheme"] == s]
        sn = len(s_rows)
        sh1 = sum(r["hit_at_1"] for r in s_rows)
        sh4 = sum(r["hit_at_4"] for r in s_rows)
        print(f"{s:<15} {sn:>8d} {sh1:>8d} {sh1 / sn * 100:>11.2f}% {sh4:>8d} {sh4 / sn * 100:>11.2f}%")
    print("-" * 65)

    # 6. Breakdown by Question Type
    print("\n" + "=" * 80)
    print("  PERFORMANCE BY QUESTION TYPE")
    print("=" * 80)
    types = sorted(list({r["type"] for r in eval_rows}))
    print(f"{'Type':<15} {'Total':>8} {'Hit@1':>8} {'Hit@1 (%)':>12} {'Hit@4':>8} {'Hit@4 (%)':>12}")
    print("-" * 65)
    for t in types:
        t_rows = [r for r in eval_rows if r["type"] == t]
        tn = len(t_rows)
        th1 = sum(r["hit_at_1"] for r in t_rows)
        th4 = sum(r["hit_at_4"] for r in t_rows)
        print(f"{t:<15} {tn:>8d} {th1:>8d} {th1 / tn * 100:>11.2f}% {th4:>8d} {th4 / tn * 100:>11.2f}%")
    print("-" * 65)

    # 7. List Every Missed Question
    missed_h4 = [r for r in eval_rows if r["hit_at_4"] == 0]
    missed_h1_only = [r for r in eval_rows if r["hit_at_1"] == 0 and r["hit_at_4"] == 1]

    print("\n" + "=" * 80)
    print(f"  MISSED QUESTIONS (Hit@4 Misses: {len(missed_h4)})")
    print("=" * 80)
    if not missed_h4:
        print("  [PERFECT HIT@4] No questions missed! All answerable questions retrieved")
        print("  the expected source file and page within +/- 1 in the top-4 chunks.")
    else:
        for idx, m in enumerate(missed_h4, start=1):
            print(f"\n[{idx}] Question ID: {m['id']} | Scheme: {m['scheme']} | Type: {m['type']}")
            print(f"    Question:  {m['question']}")
            print(f"    Expected:  {m['expected_file']} (Pages {m['expected_pages']})")
            print(f"    Retrieved: {m['retrieved_pages_summary']}")

    if missed_h1_only:
        print("\n" + "-" * 80)
        print(f"  RETRIEVED AT RANK 2-4 (Missed Hit@1, but Hit@4 Success: {len(missed_h1_only)})")
        print("-" * 80)
        for idx, m in enumerate(missed_h1_only, start=1):
            print(f"[{idx}] {m['id']} (Rank {m['hit_rank']}): {m['question'][:75]}...")
            print(f"    Expected:  {m['expected_file']} (Pages {m['expected_pages']})")
            print(f"    Retrieved: {m['retrieved_pages']}")

    # 8. Save results to results/retrieval_eval.csv
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "retrieval_eval.csv"

    fieldnames = [
        "id",
        "question",
        "scheme",
        "type",
        "expected_file",
        "expected_pages",
        "hit_at_1",
        "hit_at_4",
        "hit_rank",
        "retrieved_pages",
        "retrieved_scores",
        "retrieved_chunk_ids",
        "retrieved_pages_summary",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_rows)

    print("\n" + "=" * 80)
    print(f"Saved evaluation results to: {csv_path}")
    print("=" * 80)


def main():
    run_retrieval_evaluation()


if __name__ == "__main__":
    main()
