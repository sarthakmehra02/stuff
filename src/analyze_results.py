"""Summarize judged benchmark output with paired tests, bootstrap CIs, and plots."""

import argparse
import csv
import itertools
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = ("zero_shot", "prompted", "rag", "finetuned", "finetuned_rag")


def bootstrap_ci(values: list[bool | float], iterations: int = 1000, seed: int = 42):
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(values)
    stats = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations)]
    return tuple(float(x) for x in np.percentile(stats, [2.5, 97.5]))


def mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    if len(a) != len(b):
        raise ValueError("Paired outcomes must have equal lengths")
    discord_a = sum(x and not y for x, y in zip(a, b))
    discord_b = sum(y and not x for x, y in zip(a, b))
    n = discord_a + discord_b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(discord_a, discord_b) + 1)) / (2 ** n))
    return discord_a, discord_b, p


def _load(results_dir: Path):
    loaded = {}
    for condition in CONDITIONS:
        raw_path, judge_path = results_dir / f"raw_{condition}.jsonl", results_dir / f"judged_{condition}.jsonl"
        if not raw_path.exists() or not judge_path.exists():
            continue
        raw = {r["id"]: r for r in (json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip())}
        judged = {r["id"]: r for r in (json.loads(line) for line in judge_path.read_text(encoding="utf-8").splitlines() if line.strip())}
        loaded[condition] = {qid: {**row, **judged[qid]} for qid, row in raw.items() if qid in judged}
    return loaded


def analyze(results_dir: Path = ROOT / "results", test_path: Path = ROOT / "data" / "test_qa.json"):
    loaded = _load(results_dir)
    test = json.loads(test_path.read_text(encoding="utf-8"))
    test_by_id = {row["id"]: row for row in test}
    if not loaded:
        raise ValueError("No matching raw and judged result files were found")
    answerable_ids = [row["id"] for row in test if row["answerable"]]
    unanswerable_ids = [row["id"] for row in test if not row["answerable"]]

    def metrics(records, ids, seed):
        selected = [records[qid] for qid in ids if qid in records]
        correct = [row["verdict"] == "correct" for row in selected]
        lenient = [row["verdict"] in {"correct", "partial"} for row in selected]
        ci_strict, ci_lenient = bootstrap_ci(correct, seed=seed), bootstrap_ci(lenient, seed=seed + 1)
        return {"n": len(selected), "strict_accuracy": sum(correct) / len(correct) if correct else math.nan,
                "strict_ci_low": ci_strict[0], "strict_ci_high": ci_strict[1],
                "lenient_accuracy": sum(lenient) / len(lenient) if lenient else math.nan,
                "lenient_ci_low": ci_lenient[0], "lenient_ci_high": ci_lenient[1]}

    overall_rows = []
    for ci, (condition, records) in enumerate(loaded.items()):
        answer_metrics = metrics(records, answerable_ids, 42 + ci * 11)
        unans = [records[qid] for qid in unanswerable_ids if qid in records]
        ans = [records[qid] for qid in answerable_ids if qid in records]
        rag_rows = [row for row in records.values() if row.get("condition") in {"rag", "finetuned_rag"}]
        latencies = [float(row.get("latency_s", 0)) for row in records.values() if row.get("latency_s") is not None]
        tokens = [int(row.get("new_tokens", 0)) for row in records.values()]
        overall_rows.append({
            "condition": condition, **answer_metrics,
            "hallucination_rate": sum(r["verdict"] == "hallucinated" for r in unans) / len(unans) if unans else math.nan,
            "false_refusal_rate": sum(bool(r.get("is_refusal")) for r in ans) / len(ans) if ans else math.nan,
            "faithfulness_rate": (sum(r.get("faithfulness") == "supported" for r in rag_rows) / len(rag_rows)) if rag_rows else math.nan,
            "mean_latency_s": float(np.mean(latencies)) if latencies else math.nan,
            "p95_latency_s": float(np.percentile(latencies, 95)) if latencies else math.nan,
            "mean_new_tokens": float(np.mean(tokens)) if tokens else math.nan,
        })
    _write_csv(results_dir / "summary_overall.csv", overall_rows)

    def grouped_summary(field):
        groups = sorted({row.get(field, "") for row in test if row.get("answerable")})
        output = []
        for group in groups:
            ids = [row["id"] for row in test if row.get("answerable") and row.get(field) == group]
            for ci, (condition, records) in enumerate(loaded.items()):
                output.append({field: group, "condition": condition, **metrics(records, ids, 42 + ci * 11)})
        return output
    _write_csv(results_dir / "summary_by_type.csv", grouped_summary("type"))
    _write_csv(results_dir / "summary_by_scheme.csv", grouped_summary("scheme"))

    paired = []
    conds = list(loaded)
    for left, right in itertools.combinations(conds, 2):
        ids = [qid for qid in answerable_ids if qid in loaded[left] and qid in loaded[right]]
        a = [loaded[left][qid]["verdict"] == "correct" for qid in ids]
        b = [loaded[right][qid]["verdict"] == "correct" for qid in ids]
        x, y, p = mcnemar_exact(a, b)
        paired.append({"condition_a": left, "condition_b": right, "n": len(ids), "b_a_correct_b_wrong": x,
                       "c_a_wrong_b_correct": y, "exact_two_sided_p": p})
    _write_csv(results_dir / "paired_tests.csv", paired, comment="# McNemar exact test; n=51 is small.\n")

    retrieval = []
    for condition in ("rag", "finetuned_rag"):
        if condition not in loaded:
            continue
        counts = {(hit, correct): 0 for hit in (False, True) for correct in (False, True)}
        for qid in answerable_ids:
            row = loaded[condition].get(qid)
            if row is not None and row.get("retrieval_hit") is not None:
                counts[(bool(row["retrieval_hit"]), row["verdict"] == "correct")] += 1
        for (hit, correct), count in counts.items():
            retrieval.append({"condition": condition, "retrieval_hit": hit, "answer_correct": correct, "count": count})
    _write_csv(results_dir / "retrieval_vs_correctness.csv", retrieval)

    errors = ["# Error analysis", ""]
    for condition, records in loaded.items():
        errors += [f"## {condition}", ""]
        cases = [records[qid] for qid in test_by_id if qid in records and
                 (records[qid]["verdict"] in {"wrong", "hallucinated"})]
        for row in cases[:10]:
            expected_scheme = test_by_id[row["id"]].get("scheme", "")
            cited = re.findall(r"\[([^\]]+),\s*Page\s+\d+\]", row.get("answer", ""), flags=re.I)
            wrong_citation = any(expected_scheme.lower().replace("-", "") not in source.lower().replace("-", "") for source in cited)
            errors += [f"- **{row['id']}** — wrong scheme citation/use: {wrong_citation}",
                       f"  - Question: {row.get('question', '')}", f"  - Gold: {row.get('gold_answer', '')}",
                       f"  - Answer: {row.get('answer', '')}", f"  - Judge: {row.get('reason', '')}"]
        if not cases:
            errors += ["No wrong or hallucinated cases in available judged rows."]
        errors.append("")
    (results_dir / "error_analysis.md").write_text("\n".join(errors), encoding="utf-8")
    _plots(results_dir, overall_rows, loaded, test)
    return overall_rows


def _write_csv(path: Path, rows: list[dict], comment: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        if comment:
            stream.write(comment)
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def _plots(results_dir: Path, overall, loaded, test):
    plots = results_dir / "plots"; plots.mkdir(parents=True, exist_ok=True)
    names = [row["condition"] for row in overall]
    acc = [row["strict_accuracy"] for row in overall]
    low = [max(0, value - row["strict_ci_low"]) for value, row in zip(acc, overall)]
    high = [max(0, row["strict_ci_high"] - value) for value, row in zip(acc, overall)]
    plt.figure(figsize=(8, 4)); plt.bar(names, acc, yerr=[low, high], capsize=4); plt.ylim(0, 1); plt.ylabel("Strict accuracy"); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(plots / "accuracy_by_condition.png", dpi=160); plt.close()
    plt.figure(figsize=(8, 4)); plt.bar(names, [row["hallucination_rate"] for row in overall]); plt.ylim(0, 1); plt.ylabel("Hallucination rate"); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(plots / "hallucination_rate.png", dpi=160); plt.close()
    plt.figure(figsize=(8, 4)); plt.bar(names, [row["mean_latency_s"] for row in overall]); plt.ylabel("Mean latency (seconds)"); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(plots / "latency_by_condition.png", dpi=160); plt.close()
    types = sorted({row["type"] for row in test if row["answerable"]})
    matrix = [[sum(loaded[c].get(row["id"], {}).get("verdict") == "correct" for row in test if row["answerable"] and row["type"] == kind) / max(1, sum(row["answerable"] and row["type"] == kind for row in test)) for kind in types] for c in names]
    plt.figure(figsize=(9, 4)); plt.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto"); plt.xticks(range(len(types)), types, rotation=25); plt.yticks(range(len(names)), names); plt.colorbar(label="Strict accuracy"); plt.tight_layout(); plt.savefig(plots / "accuracy_heatmap.png", dpi=160); plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    for row in analyze(args.results):
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
