"""Blind NIM judging for model answers, with resumable and idempotent outputs."""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import JUDGE_MODEL, PATHS, REFUSAL_STRING

PROMPT_VERSION = "judge-v1"
ANSWER_PROMPT = """Grade the answer without guessing which system produced it. You see only a question, a reference answer, and a candidate answer.
For answerable questions, use verdict correct if all key facts in the reference are present and nothing contradicts it, partial if some key facts are missing/wrong, and wrong otherwise.
For unanswerable questions, use refused if the candidate says the information is unavailable and gives no specific answer; otherwise use hallucinated.
Return only JSON: {\"verdict\":\"...\",\"reason\":\"one sentence\"}.

Question: {question}
Reference answer: {gold}
Candidate answer: {answer}
Answerable: {answerable}"""
FAITHFUL_PROMPT = """Judge whether the candidate answer is supported by the supplied retrieved excerpts. Return only JSON with verdict supported, partial, or unsupported and a one-sentence reason.
Question: {question}
Candidate answer: {answer}
Retrieved excerpts: {context}"""


def normalize(text: str) -> str:
    return " ".join(str(text or "").lower().split()).strip(" .!\t\n")


def is_refusal(text: str) -> bool:
    return normalize(text) == normalize(REFUSAL_STRING)


def extract_json_object(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?|```", "", text, flags=re.I).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[index:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
    return None


def cache_key(model: str, qid: str, answer: str, prompt_version: str = PROMPT_VERSION) -> str:
    digest = hashlib.sha256(answer.encode("utf-8")).hexdigest()
    return hashlib.sha256(f"{model}|{prompt_version}|{qid}|{digest}".encode("utf-8")).hexdigest()


def _cached_call(client, cache: dict, key: str, prompt: str) -> dict:
    if key in cache:
        return cache[key]
    for attempt in range(3):
        user = prompt if attempt == 0 else prompt + f"\nReturn valid JSON only. Retry {attempt}."
        text, _, _ = client.generate(messages=[{"role": "user", "content": user}], model=JUDGE_MODEL,
                                     temperature=0, max_tokens=2048, use_cache=True)
        result = extract_json_object(text)
        if result:
            cache[key] = result
            return result
    raise ValueError("Judge returned invalid JSON after three attempts")


def _retrieval_hit(row: dict, gold: dict) -> bool | None:
    retrieved = row.get("retrieved", [])
    if not retrieved:
        return None
    expected_file = gold.get("source_file")
    pages = gold.get("source_pages") or []
    return any(item.get("source_file") == expected_file and any(
        abs(int(item.get("page", -1000)) - int(page)) <= 1 for page in pages
    ) for item in retrieved)


def run_judging(conditions: list[str], limit: int | None = None, results_dir: Path = PATHS.results_dir,
                client=None, seed: int = 42) -> dict[str, Path]:
    from src.nim_client import NIMClient
    client = client or NIMClient()
    cache_file = PATHS.cache_dir / "judge_cache.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    test_rows = json.loads((PATHS.project_root / "data" / "test_qa.json").read_text(encoding="utf-8"))
    gold_by_id = {row["id"]: row for row in test_rows}
    chunk_path = PATHS.project_root / "data" / "processed" / "chunks.json"
    chunk_map = {row["id"]: row for row in json.loads(chunk_path.read_text(encoding="utf-8"))}
    work = []
    for condition in conditions:
        source = results_dir / f"raw_{condition}.jsonl"
        target = results_dir / f"judged_{condition}.jsonl"
        if not source.exists():
            raise FileNotFoundError(f"Missing model output: {source}")
        already = set()
        if target.exists():
            for line in target.read_text(encoding="utf-8").splitlines():
                try:
                    already.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if limit is not None:
            rows = rows[:limit]
        work.extend((condition, row, target) for row in rows if row["id"] not in already)
    random.Random(seed).shuffle(work)
    opened = {}
    try:
        for condition, row, target in work:
            gold = gold_by_id[row["id"]]
            answer = row.get("answer", "")
            key = cache_key(JUDGE_MODEL, row["id"], answer)
            judgement = _cached_call(client, cache, key, ANSWER_PROMPT.format(
                question=row["question"], gold=row["gold_answer"], answer=answer,
                answerable=str(bool(row.get("answerable", gold.get("answerable")))).lower()))
            verdict = str(judgement.get("verdict", "wrong")).lower()
            allowed = {"correct", "partial", "wrong"} if gold["answerable"] else {"refused", "hallucinated"}
            if verdict not in allowed:
                verdict = "wrong" if gold["answerable"] else "hallucinated"
            faithfulness = ""
            if condition in {"rag", "finetuned_rag"}:
                retrieved_chunks = [chunk_map[item["chunk_id"]]["text"] for item in row.get("retrieved", [])
                                    if item.get("chunk_id") in chunk_map]
                context = "\n\n".join(retrieved_chunks)
                fkey = cache_key(JUDGE_MODEL, row["id"], answer + context, "faithfulness-v1")
                faith = _cached_call(client, cache, fkey, FAITHFUL_PROMPT.format(
                    question=row["question"], answer=answer, context=context))
                faithfulness = faith.get("verdict", "unsupported")
                if faithfulness not in {"supported", "partial", "unsupported"}:
                    faithfulness = "unsupported"
            result = {
                "id": row["id"], "condition": condition, "verdict": verdict,
                "reason": str(judgement.get("reason", "")), "is_refusal": is_refusal(answer),
                "retrieval_hit": _retrieval_hit(row, gold) if condition in {"rag", "finetuned_rag"} else None,
                "faithfulness": faithfulness,
            }
            if target not in opened:
                target.parent.mkdir(parents=True, exist_ok=True)
                opened[target] = target.open("a", encoding="utf-8", newline="\n")
            opened[target].write(json.dumps(result, ensure_ascii=False) + "\n")
            opened[target].flush()
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    finally:
        for stream in opened.values():
            stream.close()
    return {condition: results_dir / f"judged_{condition}.jsonl" for condition in conditions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    conditions = [part.strip() for part in args.conditions.split(",") if part.strip()]
    run_judging(conditions, args.limit)


if __name__ == "__main__":
    main()
