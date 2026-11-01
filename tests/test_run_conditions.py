import json
from pathlib import Path

import pytest

from src.run_conditions import completed_ids, run_conditions


class FakeModel:
    pass


class FakeTokenizer:
    pass


def fake_generate(model, tokenizer, messages, use_adapter, max_new_tokens):
    return {"text": "local fake answer", "latency_s": 0.01, "prompt_tokens": 7, "new_tokens": 3}


def test_adapter_required():
    with pytest.raises(ValueError, match="--adapter"):
        run_conditions(["finetuned"], model=FakeModel(), tokenizer=FakeTokenizer(), generate_fn=fake_generate)


def test_resume_skips_completed_and_writes_schema(tmp_path):
    out = tmp_path / "out"
    target = out / "raw_zero_shot.jsonl"
    target.parent.mkdir()
    test_rows = json.loads((Path(__file__).parents[1] / "data" / "test_qa.json").read_text(encoding="utf-8"))[:3]
    target.write_text(json.dumps({"id": test_rows[0]["id"], "condition": "zero_shot", "question": test_rows[0]["question"],
                                  "gold_answer": test_rows[0]["gold_answer"], "type": test_rows[0]["type"],
                                  "scheme": test_rows[0]["scheme"], "answerable": test_rows[0]["answerable"],
                                  "answer": "done", "latency_s": 0.1, "prompt_tokens": 1, "new_tokens": 1,
                                  "retrieved": [], "model_id": "test", "adapter_path": None, "timestamp": "now"}) + "\n", encoding="utf-8")
    outputs = run_conditions(["zero_shot"], limit=3, out=out, resume=True,
                             model=FakeModel(), tokenizer=FakeTokenizer(), generate_fn=fake_generate)
    records = [json.loads(line) for line in outputs["zero_shot"].read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in records] == [test_rows[0]["id"], test_rows[1]["id"], test_rows[2]["id"]]
    expected = {"condition", "question", "gold_answer", "type", "scheme", "answerable", "answer",
                "latency_s", "prompt_tokens", "new_tokens", "retrieved", "model_id", "adapter_path", "timestamp"}
    assert expected <= records[1].keys()
    assert records[1]["retrieved"] == []
    assert completed_ids(target) == {row["id"] for row in test_rows}


def test_rag_records_retrieval_metadata(tmp_path):
    chunks = [{"id": "c1", "source_file": "guide.pdf", "page": 4, "score": 0.9, "text": "context"}]
    outputs = run_conditions(["rag"], limit=1, out=tmp_path, model=FakeModel(), tokenizer=FakeTokenizer(),
                             retrieve_fn=lambda question, k: chunks, generate_fn=fake_generate)
    row = json.loads(outputs["rag"].read_text(encoding="utf-8").splitlines()[0])
    assert row["retrieved"] == [{"chunk_id": "c1", "source_file": "guide.pdf", "page": 4, "score": 0.9}]
