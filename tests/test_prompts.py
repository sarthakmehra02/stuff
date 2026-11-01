import json
from pathlib import Path

import numpy as np
import pytest

from config import REFUSAL_STRING, SYSTEM_BASE
from src.prompts import build_messages, load_fewshot, validate_fewshot

ROOT = Path(__file__).resolve().parents[1]


def test_exact_base_prompt_and_zero_shot_structure():
    assert SYSTEM_BASE == ("You answer questions about three Indian farmer schemes: PM-KISAN (income support), PMFBY (crop insurance) and the Kisan Credit Card (crop loans and interest subvention). Answer in 1-3 short sentences. If the answer is not in the official documents or you are not sure, reply exactly: Not found in the documents")
    assert build_messages("zero_shot", "Question?") == [{"role": "user", "content": "Question?"}]


def test_prompted_and_finetuned_structures():
    examples = [{"question": "Example?", "answer": "Example."}] * 4
    prompted = build_messages("prompted", "Test?", fewshot=examples)
    assert prompted[0] == {"role": "system", "content": SYSTEM_BASE}
    assert [message["role"] for message in prompted] == ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant", "user"]
    finetuned = build_messages("finetuned", "Test?")
    assert finetuned == [{"role": "system", "content": SYSTEM_BASE}, {"role": "user", "content": "Test?"}]


def test_four_examples_are_document_grounded_and_not_exact_test_questions():
    examples = load_fewshot()
    test = json.loads((ROOT / "data" / "test_qa.json").read_text(encoding="utf-8"))
    chunks = json.loads((ROOT / "data" / "processed" / "chunks.json").read_text(encoding="utf-8"))
    corpus = " ".join(chunk["text"] for chunk in chunks).lower()
    assert len(examples) == 4
    for phrase in ("national payments corporation of india (npci)", "headed by chief executive officer (ceo)", "2 years relevant experience"):
        assert phrase in corpus
    test_questions = {row["question"].strip().lower() for row in test}
    assert all(example["question"].strip().lower() not in test_questions for example in examples)
    assert examples[-1]["answer"] == REFUSAL_STRING


def test_rag_conditions_require_context():
    with pytest.raises(ValueError):
        build_messages("rag", "Question?")


def test_fewshot_validator_uses_strict_cosine_threshold():
    class OrthogonalEmbedder:
        def encode(self, texts, **kwargs):
            return np.eye(len(texts), dtype=np.float32)
    scores = validate_fewshot(OrthogonalEmbedder())
    assert len(scores) == 4 and max(scores) < 0.85
