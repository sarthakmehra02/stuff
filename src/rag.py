"""RAG Retrieval Pipeline using BAAI/bge-small-en-v1.5 and FAISS.

This module provides semantic vector retrieval over farmer scheme chunks
(PM-KISAN, PMFBY, KCC) using:
- SentenceTransformer model: BAAI/bge-small-en-v1.5 running on CPU.
- FAISS IndexFlatIP: Inner product over L2-normalized embeddings,
  mathematically equivalent to exact cosine similarity.
- Scheme-level filtering: Optional metadata filter for 'PM-KISAN', 'PMFBY', 'KCC'.
- build_rag_prompt(): Formats retrieved chunks into a factual, strictly-grounded
  prompt that requires file + page citations and a strict 'Not found in the documents'
  fallback when unanswerable.

CLI Usage:
    python src/rag.py "who is eligible for PM-KISAN?"
    python src/rag.py "what is the scale of finance?" --scheme KCC --k 4
    python src/rag.py --rebuild-index
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Reconfigure standard output for UTF-8 on Windows consoles to print currency symbols (e.g. ₹)
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PATHS, GENERATION_MODEL, ensure_directories
from src.nim_client import NIMClient


# Default embedding model and query prefix for BGE
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# In-memory model cache
_MODEL_CACHE: Dict[Tuple[str, str], SentenceTransformer] = {}


def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
) -> SentenceTransformer:
    """Load and cache the SentenceTransformer model on the requested device."""
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[key]


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_index(
    chunks_path: Optional[Path] = None,
    index_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
    batch_size: int = 32,
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Embed chunks from chunks.json and save the FAISS index, metadata, and manifest.

    Args:
        chunks_path: Path to input chunks.json (defaults to PATHS.chunks_file).
        index_path: Path to save faiss_index.bin (defaults to PATHS.faiss_index_file).
        metadata_path: Path to save faiss_metadata.json (defaults to PATHS.faiss_metadata_file).
        manifest_path: Path to save faiss_manifest.json (defaults to index_path.parent / faiss_manifest.json).
        model_name: Sentence-transformers model name.
        device: Device to run embeddings on ('cpu').
        batch_size: Embedding batch size.

    Returns:
        Tuple of (faiss_index, list_of_chunks).
    """
    ensure_directories()
    chunks_path = Path(chunks_path or PATHS.chunks_file)
    index_path = Path(index_path or PATHS.faiss_index_file)
    metadata_path = Path(metadata_path or PATHS.faiss_metadata_file)
    manifest_path = Path(manifest_path or (index_path.parent / "faiss_manifest.json"))

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at: {chunks_path}. Please run 'python src/ingest.py' first."
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}.")

    print(f"Loading embedding model '{model_name}' on {device.upper()}...")
    model = get_embedding_model(model_name=model_name, device=device)

    # Prepare texts for embedding: include scheme and clause prefix for rich semantic context
    texts_to_embed = [
        f"Scheme: {c.get('scheme', '')} | Clause: {c.get('clause_at_start') or 'General'} | {c.get('text', '')}"
        for c in chunks
    ]

    print(f"Embedding {len(texts_to_embed)} chunks (batch_size={batch_size})...")
    embeddings = model.encode(
        texts_to_embed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.ascontiguousarray(embeddings.astype("float32"))

    dim = embeddings.shape[1]
    print(f"Creating FAISS IndexFlatIP (dim={dim}, cosine similarity via normalized embeddings)...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Persist index and metadata to disk
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Save manifest with number of chunks and SHA-256 hash of chunks.json
    chunks_hash = compute_file_sha256(chunks_path)
    manifest = {
        "num_chunks": len(chunks),
        "chunks_hash": chunks_hash,
        "embedding_model": model_name,
        "index_file": index_path.name,
        "metadata_file": metadata_path.name,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved FAISS index ({index.ntotal} vectors) to: {index_path}")
    print(f"Saved chunk metadata ({len(chunks)} records) to: {metadata_path}")
    print(f"Saved manifest ({len(chunks)} chunks, sha256={chunks_hash[:12]}...) to: {manifest_path}")

    return index, chunks


def load_index(
    index_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    chunks_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    auto_build_if_missing: bool = True,
    force_rebuild: bool = False,
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Load persisted FAISS index and chunk metadata from disk.

    Verifies index freshness against chunks.json via SHA-256 manifest.
    If stale or missing, automatically rebuilds and prints 'Index was stale, rebuilt'.
    Always prints 'Index: N vectors, N chunks' and asserts equality.

    Args:
        index_path: Path to faiss_index.bin.
        metadata_path: Path to faiss_metadata.json.
        chunks_path: Path to chunks.json.
        manifest_path: Path to faiss_manifest.json.
        auto_build_if_missing: If True, builds index if files are missing.
        force_rebuild: If True, forces rebuilding the index.

    Returns:
        Tuple of (faiss_index, list_of_chunks).
    """
    index_path = Path(index_path or PATHS.faiss_index_file)
    metadata_path = Path(metadata_path or PATHS.faiss_metadata_file)
    chunks_path = Path(chunks_path or PATHS.chunks_file)
    manifest_path = Path(manifest_path or (index_path.parent / "faiss_manifest.json"))

    if force_rebuild:
        index, chunks = build_index(
            chunks_path=chunks_path,
            index_path=index_path,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
        )
    else:
        rebuild_needed = False
        if not index_path.exists() or not metadata_path.exists() or not manifest_path.exists():
            if not auto_build_if_missing:
                raise FileNotFoundError(
                    f"FAISS index, metadata, or manifest missing. Run 'python src/rag.py --rebuild'."
                )
            rebuild_needed = True
        else:
            current_hash = compute_file_sha256(chunks_path)
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                saved_hash = manifest.get("chunks_hash")
            except Exception:
                saved_hash = None

            if saved_hash != current_hash:
                rebuild_needed = True

        if rebuild_needed:
            index, chunks = build_index(
                chunks_path=chunks_path,
                index_path=index_path,
                metadata_path=metadata_path,
                manifest_path=manifest_path,
            )
            print("Index was stale, rebuilt")
        else:
            index = faiss.read_index(str(index_path))
            with open(metadata_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

    # Print 'Index: N vectors, N chunks' on every load and assert they are equal
    print(f"Index: {index.ntotal} vectors, {len(chunks)} chunks")
    assert index.ntotal == len(chunks), (
        f"Index vector count ({index.ntotal}) != chunks count ({len(chunks)})"
    )

    return index, chunks


def retrieve(
    question: str,
    k: int = 4,
    scheme: Optional[str] = None,
    index: Optional[faiss.Index] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Retrieve top-k chunks matching question with cosine similarity scores.

    Args:
        question: User query string.
        k: Number of chunks to retrieve (default: 4).
        scheme: Optional scheme name to filter by ('PM-KISAN', 'PMFBY', 'KCC').
        index: Pre-loaded FAISS index (loads from disk if None).
        chunks: Pre-loaded chunks list (loads from disk if None).
        model_name: Sentence-transformers model name.
        device: Device to run embedding model on ('cpu').

    Returns:
        List of chunk dicts, each with an added 'score' field (float cosine similarity).
    """
    if index is None or chunks is None:
        index, chunks = load_index()

    if not chunks or index.ntotal == 0:
        return []

    model = get_embedding_model(model_name=model_name, device=device)

    # BGE recommendation: prefix query with instruction for asymmetric retrieval
    query_text = f"{BGE_QUERY_PREFIX}{question.strip()}"
    query_emb = model.encode(
        [query_text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_emb = np.ascontiguousarray(query_emb.astype("float32"))

    # If scheme filter is applied, retrieve a larger candidate pool then filter
    if scheme:
        scheme_norm = scheme.strip().upper().replace("_", "-")
        candidate_k = min(len(chunks), max(k * 10, 50))
        scores, indices = index.search(query_emb, candidate_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = dict(chunks[idx])
            chunk_scheme = str(chunk.get("scheme", "")).strip().upper().replace("_", "-")
            if chunk_scheme == scheme_norm or scheme_norm in chunk_scheme:
                chunk["score"] = round(float(score), 4)
                results.append(chunk)
                if len(results) == k:
                    break
        return results

    # No filter: retrieve top-k directly
    scores, indices = index.search(query_emb, min(k, len(chunks)))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = round(float(score), 4)
        results.append(chunk)

    return results


def build_rag_prompt(
    question: str,
    chunks: List[Dict[str, Any]],
) -> str:
    """Build a grounded prompt strictly constraining the model to the context.

    Instructs the model:
      - Answer ONLY from the provided context.
      - Cite source file and page for every factual claim.
      - Reply 'Not found in the documents' if the answer is not present.

    Args:
        question: User inquiry.
        chunks: List of retrieved chunk dicts.

    Returns:
        Formatted prompt string.
    """
    context_blocks = []
    for i, c in enumerate(chunks, 1):
        source = c.get("source_file", "unknown")
        page = c.get("page", "?")
        scheme = c.get("scheme", "")
        clause = c.get("clause_at_start") or "N/A"
        text = c.get("text", "").strip()

        block = (
            f"[{i}] Source: {source} (Page {page}) | Scheme: {scheme} | Clause: {clause}\n"
            f"{text}"
        )
        context_blocks.append(block)

    context_str = "\n\n".join(context_blocks) if context_blocks else "[No context available]"

    prompt = (
        "You are an expert factual assistant specializing in Indian Government Farmer Schemes "
        "(PM-KISAN, PMFBY, KCC).\n"
        "Answer the user's question STRICTLY and ONLY using the facts stated in the context below.\n"
        "Do NOT extrapolate, assume, or incorporate outside knowledge.\n\n"
        "CRITICAL RULES:\n"
        "1. Direct Source Citations: Cite the specific source file and page number for every fact "
        "(e.g., [pmkisan_operational_guidelines_2020.pdf, Page 3]).\n"
        "2. Strict Truthfulness: If the answer cannot be found in the provided context, respond "
        "EXACTLY with:\n"
        "   Not found in the documents\n\n"
        f"Context:\n"
        f"{context_str}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return prompt


def answer(
    question: str,
    k: int = 4,
    scheme: Optional[str] = None,
    generate_fn: Optional[Any] = None,
) -> Tuple[str, List[Dict[str, Any]], str, float, bool]:
    """Perform end-to-end RAG: retrieve chunks, assemble grounded prompt, generate answer.

    Args:
        question:     User query string.
        k:            Number of chunks to retrieve (default: 4).
        scheme:       Optional scheme filter ('PM-KISAN', 'PMFBY', 'KCC').
        generate_fn:  Callable[str] -> str that accepts a prompt and returns the
                      model's reply.  When None, a default NIM closure is created
                      using config.GENERATION_MODEL (intended for local dev only;
                      callers should supply their own generate_fn in production).

    Returns:
        Tuple of (answer_text, retrieved_chunks, model_label, latency_sec, is_cached).
        - model_label: human-readable string identifying what produced the answer.
        - latency_sec / is_cached: only meaningful when the built-in NIM closure is used;
          external generate_fn callers will receive 0.0 / False respectively.
    """
    # 1. Retrieve top-k chunks
    chunks = retrieve(question, k=k, scheme=scheme)
    if not chunks:
        return "Not found in the documents", [], "none", 0.0, False

    # 2. Build strictly-grounded prompt
    prompt = build_rag_prompt(question, chunks)

    # 3. Resolve generate_fn
    model_label: str
    latency: float = 0.0
    from_cache: bool = False

    if generate_fn is not None:
        # Caller-supplied function — run it and record provenance
        model_label = getattr(generate_fn, "__name__", "external_generate_fn")
        import time as _time
        _t0 = _time.time()
        reply_text = generate_fn(prompt)
        latency = _time.time() - _t0
    else:
        # Dev-only default: use NIM client with GENERATION_MODEL from config
        nim_client = NIMClient()
        model_label = GENERATION_MODEL
        import time as _time
        _t0 = _time.time()
        reply_text, latency, from_cache = nim_client.generate(
            messages=[{"role": "user", "content": prompt}],
            model=GENERATION_MODEL,
            temperature=0.1,
            max_tokens=1024,
        )
        latency = _time.time() - _t0 if not from_cache else latency

    print(f"[RAG] Answer produced by: {model_label}")
    return reply_text.strip(), chunks, model_label, latency, from_cache


def main():
    parser = argparse.ArgumentParser(
        description="RAG retrieval and question answering for Indian Farmer Schemes."
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default=None,
        help="Question to query against farmer scheme guidelines.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of chunks to retrieve (default: 4).",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default=None,
        help="Optional scheme filter (PM-KISAN, PMFBY, KCC).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override NIM model ID (e.g. 'meta/llama-3.3-70b-instruct'). "
             "Wraps the model in a generate_fn closure; defaults to config.GENERATION_MODEL.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.1,
        help="Generation temperature when --model is specified (default: 0.1).",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Only display retrieved chunks without calling the LLM.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass SQLite response cache (only applies when --model is set or default NIM is used).",
    )
    parser.add_argument(
        "--rebuild",
        "--rebuild-index",
        action="store_true",
        dest="rebuild",
        help="Force rebuild of FAISS index from chunks.json.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Display the assembled RAG prompt in output.",
    )
    args = parser.parse_args()

    # Rebuild index mode
    if args.rebuild or args.question is None:
        print("=" * 80)
        print("REBUILDING FAISS INDEX")
        print("=" * 80)
        index, chunks = build_index()
        print(f"Index: {index.ntotal} vectors, {len(chunks)} chunks")
        assert index.ntotal == len(chunks), (
            f"Index vector count ({index.ntotal}) != chunks count ({len(chunks)})"
        )
        if args.question is None:
            print("\nIndex built successfully. Run with a question to test retrieval and answering:")
            print("  python src/rag.py \"who is eligible for PM-KISAN?\"")
            return

    # Retrieval and Question Answering
    question = args.question
    print("=" * 80)
    print(f"QUERY: '{question}'")
    if args.scheme:
        print(f"SCHEME FILTER: {args.scheme}")
    print(f"TOP-K: {args.k} | EMBEDDINGS: {DEFAULT_EMBEDDING_MODEL} (CPU)")
    print("=" * 80)

    results = retrieve(question, k=args.k, scheme=args.scheme)

    if not results:
        print("\nNo matching chunks found.")
        return

    print(f"\nRetrieved {len(results)} chunks:\n")
    for i, chunk in enumerate(results, 1):
        print("-" * 80)
        print(
            f"[{i}] Score: {chunk['score']:.4f} | Scheme: {chunk['scheme']} | "
            f"Page: {chunk['page']} | Clause: {chunk.get('clause_at_start') or 'N/A'}"
        )
        print(f"    Source: {chunk['source_file']} | ID: {chunk['id']}")
        preview = chunk["text"]
        if len(preview) > 280:
            preview = preview[:280] + "..."
        print(f"    Text:   {preview}")

    print("-" * 80)

    if args.show_prompt:
        print("\n" + "=" * 80)
        print("ASSEMBLED RAG PROMPT")
        print("=" * 80)
        prompt = build_rag_prompt(question, results)
        print(prompt)
        print("=" * 80)

    # Generate answer unless user specified --retrieve-only
    if not args.retrieve_only:
        # Build optional generate_fn override
        custom_generate_fn = None
        if args.model:
            _override_model = args.model
            _override_temp = args.temp
            _no_cache = args.no_cache
            _nim = NIMClient()

            def _model_generate_fn(prompt: str) -> str:
                """CLI-supplied model override generate_fn."""
                reply, _, _ = _nim.generate(
                    messages=[{"role": "user", "content": prompt}],
                    model=_override_model,
                    temperature=_override_temp,
                    use_cache=not _no_cache,
                )
                return reply

            _model_generate_fn.__name__ = _override_model  # label in log line
            custom_generate_fn = _model_generate_fn

        print("\nGenerating grounded answer using NVIDIA NIM...")
        reply, _, model_used, latency, from_cache = answer(
            question=question,
            k=args.k,
            scheme=args.scheme,
            generate_fn=custom_generate_fn,
        )
        print("\n" + "=" * 80)
        print(f"ANSWER (Model: {model_used} | Latency: {latency:.2f}s | Cached: {from_cache})")
        print("=" * 80)
        print(reply)
        print("=" * 80)


if __name__ == "__main__":
    main()

