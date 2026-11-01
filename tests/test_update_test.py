import json
from pathlib import Path

from src.update_test import create_update


def test_update_data_is_synthetic_and_original_is_unchanged(tmp_path, monkeypatch):
    import src.rag
    monkeypatch.setattr(src.rag, "build_index", lambda **kwargs: (None, []))
    original = Path(__file__).parents[1] / "data" / "processed" / "chunks.json"
    original_bytes = original.read_bytes()
    out = tmp_path / "update_test"
    questions = create_update(original, out)
    assert len(questions) == 10
    assert all(q["synthetic"] and "NOT A REAL AMENDMENT" in q["synthetic_update_label"] for q in questions)
    changed = json.loads((out / "chunks_modified.json").read_text(encoding="utf-8"))
    assert sum("synthetic_update_label" in chunk for chunk in changed) == 3
    assert (out / "SYNTHETIC_NOTICE.txt").exists()
    assert original.read_bytes() == original_bytes
