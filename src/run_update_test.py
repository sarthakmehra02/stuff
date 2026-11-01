"""Colab inference for synthetic chunks using the unchanged adapter and updated RAG."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_ID, TOP_K
from src.prompts import build_messages


def run(adapter: str, update_dir: Path, out_dir: Path):
    from src.models import load_model, generate
    from src.rag import load_index, retrieve

    index, chunks = load_index(index_path=update_dir / "faiss_index.bin",
                               metadata_path=update_dir / "faiss_metadata.json",
                               chunks_path=update_dir / "chunks_modified.json",
                               manifest_path=update_dir / "faiss_manifest.json")
    model, tokenizer = load_model(adapter_path=adapter)
    questions = json.loads((update_dir / "update_qa.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {c: out_dir / f"raw_update_{c}.jsonl" for c in ("rag", "finetuned")}
    for condition, path in paths.items():
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for item in questions:
                found = retrieve(item["question"], k=TOP_K, index=index, chunks=chunks) if condition == "rag" else []
                messages = build_messages("rag", item["question"], chunks=found) if condition == "rag" else build_messages("finetuned", item["question"])
                result = generate(model, tokenizer, messages, use_adapter=condition == "finetuned")
                record = {**item, "condition": condition, "answer": result["text"],
                          "latency_s": result["latency_s"], "prompt_tokens": result["prompt_tokens"],
                          "new_tokens": result["new_tokens"], "model_id": MODEL_ID,
                          "retrieved": [{"chunk_id": c["id"], "source_file": c["source_file"],
                                         "page": c["page"], "score": c["score"]} for c in found],
                          "timestamp": datetime.now(timezone.utc).isoformat()}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n"); stream.flush()
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--update-dir", type=Path, default=Path("data/update_test"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/update_test"))
    args = parser.parse_args()
    run(args.adapter, args.update_dir, args.out_dir)


if __name__ == "__main__":
    main()
