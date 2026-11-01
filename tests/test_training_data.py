import json
from pathlib import Path

from src.make_train_data import answer_is_english, exceeds_similarity, extract_json_array, normalize_ws, question_mentions_wrong_scheme, quote_in_chunk
from src.split_train import split_by_chunk
from src.finetune import format_example


class TinyTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = " ".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i >= 0)


def test_messy_teacher_json_and_evidence_normalization():
    assert extract_json_array('Here you go:\n```json\n[{"question":"Q"}]\n```') == [{"question": "Q"}]
    assert extract_json_array('Reasoning mentions [] first; final JSON is [{"question":"Q"}] trailing') == [{"question": "Q"}]
    assert extract_json_array("[]") == []
    assert quote_in_chunk("official   rule", "Text with official\nrule included")
    assert not quote_in_chunk("missing phrase", "other text")
    assert normalize_ws(" a\n b ") == "a b"


def test_similarity_filter_with_fake_normalized_embeddings():
    candidate = [1.0, 0.0]
    assert exceeds_similarity(candidate, [[0.9, 0.1]], 0.85)
    assert not exceeds_similarity(candidate, [[0.8, 0.6]], 0.85)
    assert not exceeds_similarity(candidate, [], 0.85)
    assert answer_is_english("Eligible farmers receive an interest subsidy.")
    assert not answer_is_english("Kisan fasal ki katayi ke baad benefit le sakta hai.")
    assert question_mentions_wrong_scheme("PM-KISAN ya KCC ke liye kya rule hai?", "KCC")
    assert not question_mentions_wrong_scheme("How does the KCC limit work?", "KCC")


def test_chunk_group_is_never_split(tmp_path):
    source = tmp_path / "train.jsonl"
    rows = [{"chunk_id": f"c{i}", "question": str(j)} for i in range(20) for j in range(4)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    train, val = split_by_chunk(source, tmp_path / "split")
    assert not ({row["chunk_id"] for row in train} & {row["chunk_id"] for row in val})
    assert len(train) + len(val) == len(rows)


def test_completion_only_labels_mask_prompt():
    row = format_example({"question": "What?", "answer": "The answer."}, TinyTokenizer())
    masked = [label for label in row["labels"] if label == -100]
    supervised = [label for label in row["labels"] if label != -100]
    assert masked and supervised
    assert "The answer." in TinyTokenizer().decode(supervised)
