"""Run one or more local Qwen benchmark conditions and resume JSONL outputs."""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_ID, PATHS, RESULTS_DIR, SEED, TEST_QA_FILE, TOP_K, RERANKER_MODEL, RERANK_CANDIDATES
from src.prompts import CONDITIONS, build_messages, prompt_fingerprint

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger("run_conditions")
ADAPTER_CONDITIONS = {"finetuned", "finetuned_rag"}
RAG_CONDITIONS = {"rag", "finetuned_rag"}


def completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _versions() -> dict[str, str]:
    result = {}
    for package in ("torch", "transformers", "peft", "trl", "bitsandbytes", "sentence-transformers", "faiss-cpu"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "TODO(user): commit repository before benchmark"


def _write_metadata(path: Path, fingerprint: str, adapter_path: str | None, started: str,
                    ended: str | None = None, peak_gpu_mb: float | None = None,
                    gpu_name: str | None = None) -> None:
    current = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        old = current.get("prompt_fingerprint")
        if old and old != fingerprint:
            raise RuntimeError("Prompt fingerprint changed from the first benchmark run; refusing mixed evaluation.")
    current.update({
        "prompt_fingerprint": fingerprint,
        "commit_hash": current.get("commit_hash") or _git_commit(),
        "model_id": MODEL_ID,
        "quantization": {"load_in_4bit": True, "type": "nf4", "double_quant": True, "compute_dtype": "float16"},
        "library_versions": _versions(),
        "seed": SEED,
        "adapter_path": adapter_path,
        "started_at": current.get("started_at") or started,
        "ended_at": ended,
        "peak_gpu_mb": peak_gpu_mb,
        "gpu_name": gpu_name,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_conditions(conditions: list[str], limit: int | None = None, adapter: str | None = None,
                   out: Path = RESULTS_DIR, resume: bool = False, model=None, tokenizer=None,
                   retrieve_fn=None, generate_fn=None, rerank: bool = False,
                   candidate_k: int = RERANK_CANDIDATES) -> dict[str, Path]:
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    if set(conditions) & ADAPTER_CONDITIONS and not adapter:
        raise ValueError("--adapter PATH is required for finetuned and finetuned_rag conditions")
    test_path = TEST_QA_FILE if TEST_QA_FILE.exists() else PATHS.project_root / "Data" / "test_qa.json"
    rows = json.loads(test_path.read_text(encoding="utf-8"))
    if limit is not None:
        rows = rows[:limit]
    if model is None or tokenizer is None:
        from src.models import load_model, generate
        model, tokenizer = load_model(adapter_path=adapter)
        generate_fn = generate
    if generate_fn is None:
        from src.models import generate
        generate_fn = generate
    if retrieve_fn is None and set(conditions) & RAG_CONDITIONS:
        from src.rag import build_index, retrieve
        build_index()
        if rerank:
            def retrieve_fn(question, k=TOP_K):
                return retrieve(question, k=k, rerank=True,
                                reranker_model=RERANKER_MODEL, candidate_k=candidate_k)
        else:
            retrieve_fn = retrieve

    fingerprint = prompt_fingerprint()
    started = datetime.now(timezone.utc).isoformat()
    metadata_path = out / "run_metadata.json"
    _write_metadata(metadata_path, fingerprint, adapter, started)
    outputs = {}
    for condition in conditions:
        output_path = out / f"raw_{condition}.jsonl"
        done = completed_ids(output_path) if resume else set()
        outputs[condition] = output_path
        mode = "a" if resume else "w"
        with output_path.open(mode, encoding="utf-8", newline="\n") as stream:
            for item in rows:
                if item["id"] in done:
                    continue
                retrieved = retrieve_fn(item["question"], k=TOP_K) if condition in RAG_CONDITIONS else []
                messages = build_messages(condition, item["question"], chunks=retrieved if condition in RAG_CONDITIONS else None)
                result = generate_fn(model, tokenizer, messages, use_adapter=condition in ADAPTER_CONDITIONS,
                                     max_new_tokens=300)
                retrieved_metadata = []
                for chunk in retrieved:
                    metadata = {"chunk_id": chunk.get("id"),
                                "source_file": chunk.get("source_file"),
                                "page": chunk.get("page"), "score": chunk.get("score")}
                    # Keep the original output schema for ordinary retrieval;
                    # add diagnostics only when reranking produced them.
                    if chunk.get("embedding_score") is not None:
                        metadata["embedding_score"] = chunk["embedding_score"]
                    if chunk.get("rerank_score") is not None:
                        metadata["rerank_score"] = chunk["rerank_score"]
                    retrieved_metadata.append(metadata)
                record = {
                    "id": item["id"], "condition": condition, "question": item["question"],
                    "gold_answer": item["gold_answer"], "type": item["type"], "scheme": item["scheme"],
                    "answerable": item["answerable"], "answer": result["text"],
                    "latency_s": result["latency_s"], "prompt_tokens": result["prompt_tokens"],
                    "new_tokens": result["new_tokens"],
                    "retrieved": retrieved_metadata,
                    "model_id": MODEL_ID, "adapter_path": adapter if condition in ADAPTER_CONDITIONS else None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                logger.info("%s %s (%.2fs): %s", condition, item["id"], result["latency_s"], result["text"])
    from src.models import peak_gpu_mb, gpu_name
    _write_metadata(metadata_path, fingerprint, adapter, started,
                    datetime.now(timezone.utc).isoformat(), peak_gpu_mb(), gpu_name())
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", required=True, help="Comma-separated benchmark conditions")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--adapter")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerank", action="store_true",
                        help="Use cross-encoder reranking for RAG conditions.")
    parser.add_argument("--candidate-k", type=int, default=RERANK_CANDIDATES)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_conditions([part.strip() for part in args.conditions.split(",") if part.strip()],
                   args.limit, args.adapter, args.out, args.resume,
                   rerank=args.rerank, candidate_k=args.candidate_k)


if __name__ == "__main__":
    main()
