"""Shared condition prompts and reproducible prompt fingerprint."""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FEWSHOT_FILE, MODEL_ID, MAX_NEW_TOKENS, REFUSAL_STRING, SEED, SYSTEM_BASE, TOP_K

CONDITIONS = ("zero_shot", "prompted", "rag", "finetuned", "finetuned_rag")


def load_fewshot(path: Path = FEWSHOT_FILE) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as stream:
        items = json.load(stream)
    if len(items) != 4 or any(not item.get("question") or not item.get("answer") for item in items):
        raise ValueError("fewshot.json must contain exactly four question/answer objects")
    return items


def build_messages(condition: str, question: str, fewshot=None, chunks=None) -> list[dict[str, str]]:
    """Build the canonical chat messages for one benchmark condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")
    if condition == "zero_shot":
        return [{"role": "user", "content": question}]
    if condition in {"rag", "finetuned_rag"}:
        if chunks is None:
            raise ValueError(f"{condition} requires retrieved chunks")
        from src.rag import build_rag_prompt
        return [{"role": "user", "content": build_rag_prompt(question, chunks)}]

    messages = [{"role": "system", "content": SYSTEM_BASE}]
    if condition == "prompted":
        for example in (load_fewshot() if fewshot is None else fewshot):
            messages.extend([
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]},
            ])
    messages.append({"role": "user", "content": question})
    return messages


def prompt_fingerprint() -> str:
    """Hash all experiment prompt inputs, including the existing RAG template."""
    from inspect import getsource
    from src.rag import build_rag_prompt
    from src.make_train_data import SYSTEM_PROMPT as TEACHER_SYSTEM, USER_PROMPT_TEMPLATE as TEACHER_TEMPLATE
    from src.judge import ANSWER_PROMPT, FAITHFUL_PROMPT

    payload = {
        "system_base": SYSTEM_BASE,
        "refusal": REFUSAL_STRING,
        "condition_builder": getsource(build_messages),
        "rag_builder": getsource(build_rag_prompt),
        "teacher_prompts": [TEACHER_SYSTEM, TEACHER_TEMPLATE],
        "judge_prompts": [ANSWER_PROMPT, FAITHFUL_PROMPT],
        "fewshot": FEWSHOT_FILE.read_text(encoding="utf-8"),
        "constants": {"model_id": MODEL_ID, "max_new_tokens": MAX_NEW_TOKENS, "seed": SEED, "top_k": TOP_K},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_fewshot(embedder=None) -> list[float]:
    """Assert examples are grounded and below the fixed test-question similarity threshold."""
    import numpy as np
    tests = json.loads((FEWSHOT_FILE.parent / "test_qa.json").read_text(encoding="utf-8"))
    chunks_path = FEWSHOT_FILE.parent / "processed" / "chunks.json"
    corpus = " ".join(row["text"] for row in json.loads(chunks_path.read_text(encoding="utf-8"))).lower()
    examples = load_fewshot()
    required = (
        "national payments corporation of india (npci)",
        "headed by chief executive officer (ceo)",
        "2 years relevant experience",
    )
    if not all(phrase in corpus for phrase in required):
        raise AssertionError("A required few-shot fact could not be verified in chunks.json")
    test_questions = [row["question"] for row in tests]
    if any(example["question"].strip().casefold() in {q.strip().casefold() for q in test_questions} for example in examples):
        raise AssertionError("A few-shot question duplicates a test-set question")
    if embedder is None:
        from src.rag import get_embedding_model
        embedder = get_embedding_model()
    all_embeddings = embedder.encode([item["question"] for item in examples] + test_questions,
                                     normalize_embeddings=True, show_progress_bar=False)
    similarities = (all_embeddings[:len(examples)] @ all_embeddings[len(examples):].T).max(axis=1)
    scores = [float(value) for value in similarities]
    if any(score >= 0.85 for score in scores):
        raise AssertionError(f"Few-shot question similarity must be < 0.85; got {scores}")
    return scores


if __name__ == "__main__":
    print(json.dumps({"max_similarity": max(validate_fewshot())}, indent=2))
