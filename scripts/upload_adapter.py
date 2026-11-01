"""Upload a trained LoRA adapter to the Hugging Face Hub using HF_TOKEN."""

import argparse
import json
import os
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def build_model_card(train_data: Path, training_summary: Path | None = None) -> str:
    count = sum(1 for line in train_data.read_text(encoding="utf-8").splitlines() if line.strip()) if train_data.exists() else "TODO(user)"
    summary = json.loads(training_summary.read_text(encoding="utf-8")) if training_summary and training_summary.exists() else {}
    return f"""---
base_model: {BASE_MODEL}
library_name: peft
tags:
- qlora
- indian-agriculture
---

# Indian Farmer Schemes QLoRA Adapter

This repository contains a LoRA adapter trained for a term-project benchmark on PM-KISAN, PMFBY, and Kisan Credit Card documents.

## Base model

`{BASE_MODEL}`. The base model weights are unchanged; training used 4-bit NF4 QLoRA with fp16 computation.

## Training data and settings

- Synthetic question/answer pairs: {count}
- Epochs: {summary.get('epochs', 'TODO(user)')}
- Steps: {summary.get('steps', 'TODO(user)')}
- Training data source: evidence-checked examples generated from official guideline chunks; no evaluation questions are included.

## Intended use

Educational demonstration and research comparison on questions about these three schemes. Use the official documents for verification.

## Limitations

Answers reflect documents dated 2018–2026, not current law or legal advice. The model may omit details or answer incorrectly. The adapter was trained without retrieval context; RAG is a separate evaluation condition.
"""


def upload(adapter: Path, repo_id: str, train_data: Path, training_summary: Path | None = None, private: bool = True):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN in the environment before uploading; the token is never written to disk.")
    if not adapter.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter}")
    from huggingface_hub import HfApi
    card = build_model_card(train_data, training_summary)
    (adapter / "README.md").write_text(card, encoding="utf-8")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    return api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(adapter), commit_message="Upload farmer schemes LoRA adapter")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="Hugging Face account/name")
    parser.add_argument("--train-data", type=Path, default=Path("data/train_qa.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/run1/training_summary.json"))
    parser.add_argument("--public", action="store_true", help="Create a public model repository (default is private)")
    args = parser.parse_args()
    print(upload(args.adapter, args.repo_id, args.train_data, args.summary, private=not args.public))


if __name__ == "__main__":
    main()
