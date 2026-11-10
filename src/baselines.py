"""Baseline evaluation conditions for Indian Farmer Schemes benchmark.

Implements three inference baselines:
1. zero_shot: Bare question, no system prompt or few-shot context.
2. prompted: System prompt + 4 few-shot examples loaded from data/fewshot.json.
3. rag_answer: RAG pipeline with top-k retrieval and local Qwen generation.

All conditions return a standardized dictionary:
{
    "answer": str,
    "latency": float,
    "retrieved_ids": List[str]
}
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEWSHOT_PATH = PROJECT_ROOT / "data" / "fewshot.json"

PROMPTED_SYSTEM_PROMPT = (
    "You answer questions about Indian farmer schemes (PM-KISAN, PMFBY, Kisan Credit Card); "
    "answer briefly; if you are not sure or the information is not in your knowledge, "
    "reply exactly 'Not found in the documents'."
)

# Global model/tokenizer singleton cache
_CACHED_MODEL = None
_CACHED_TOKENIZER = None
_FEWSHOT_CACHE = None


def get_default_model() -> Tuple[Any, Any]:
    """Retrieve or lazily initialize the default 4-bit NF4 Qwen model & tokenizer."""
    global _CACHED_MODEL, _CACHED_TOKENIZER
    if _CACHED_MODEL is None or _CACHED_TOKENIZER is None:
        from src.models import load_model
        _CACHED_MODEL, _CACHED_TOKENIZER = load_model()
    return _CACHED_MODEL, _CACHED_TOKENIZER


def set_default_model(model: Any, tokenizer: Any) -> None:
    """Manually register pre-loaded model and tokenizer."""
    global _CACHED_MODEL, _CACHED_TOKENIZER
    _CACHED_MODEL = model
    _CACHED_TOKENIZER = tokenizer


def load_fewshot_examples() -> List[Dict[str, str]]:
    """Load the 4 few-shot examples from data/fewshot.json."""
    global _FEWSHOT_CACHE
    if _FEWSHOT_CACHE is None:
        if not FEWSHOT_PATH.exists():
            raise FileNotFoundError(f"Few-shot examples file not found at {FEWSHOT_PATH}")
        with open(FEWSHOT_PATH, "r", encoding="utf-8") as f:
            _FEWSHOT_CACHE = json.load(f)
    return _FEWSHOT_CACHE


def zero_shot(
    question: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    max_new_tokens: int = 300,
) -> Dict[str, Any]:
    """Bare question with no system prompt or context.

    Args:
        question: User query string.
        model: Optional pre-loaded model.
        tokenizer: Optional pre-loaded tokenizer.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        {"answer": str, "latency": float, "retrieved_ids": []}
    """
    from src.models import generate

    if model is None or tokenizer is None:
        m, tok = get_default_model()
    else:
        m, tok = model, tokenizer

    result = generate(
        model=m,
        tokenizer=tok,
        prompt=question,
        system=None,
        max_new_tokens=max_new_tokens,
        use_adapter=False,
    )
    return {
        "answer": result["text"],
        "latency": round(result["latency_s"], 4),
        "retrieved_ids": [],
    }


def prompted(
    question: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    max_new_tokens: int = 300,
) -> Dict[str, Any]:
    """Few-shot prompted baseline with system prompt and 4 exemplars.

    Args:
        question: User query string.
        model: Optional pre-loaded model.
        tokenizer: Optional pre-loaded tokenizer.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        {"answer": str, "latency": float, "retrieved_ids": []}
    """
    from src.models import generate

    if model is None or tokenizer is None:
        m, tok = get_default_model()
    else:
        m, tok = model, tokenizer

    examples = load_fewshot_examples()

    # Build multi-turn chat messages
    messages = [{"role": "system", "content": PROMPTED_SYSTEM_PROMPT}]
    for ex in examples:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": question})

    result = generate(
        model=m,
        tokenizer=tok,
        messages=messages,
        max_new_tokens=max_new_tokens,
        use_adapter=False,
    )
    return {
        "answer": result["text"],
        "latency": round(result["latency_s"], 4),
        "retrieved_ids": [],
    }


def rag_answer(
    question: str,
    k: int = 4,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    scheme: Optional[str] = None,
    max_new_tokens: int = 300,
    rerank: bool = False,
    candidate_k: int = 20,
) -> Dict[str, Any]:
    """Retrieval-augmented baseline using top-k semantic retrieval and local Qwen generation.

    Args:
        question: User query string.
        k: Number of chunks to retrieve (default: 4).
        model: Optional pre-loaded model.
        tokenizer: Optional pre-loaded tokenizer.
        scheme: Optional scheme filter.
        max_new_tokens: Maximum tokens to generate.
        rerank: Apply cross-encoder reranking to the semantic candidate pool.
        candidate_k: Number of embedding candidates scored by the reranker.

    Returns:
        {"answer": str, "latency": float, "retrieved_ids": List[str]}
    """
    from src.models import generate
    from src.rag import answer as rag_pipeline_answer

    if model is None or tokenizer is None:
        m, tok = get_default_model()
    else:
        m, tok = model, tokenizer

    def qwen_generate_fn(rag_prompt: str) -> str:
        result = generate(
            model=m,
            tokenizer=tok,
            prompt=rag_prompt,
            system=None,
            max_new_tokens=max_new_tokens,
            use_adapter=False,
        )
        return result["text"]

    reply_text, chunks, _label, latency, _from_cache = rag_pipeline_answer(
        question=question,
        k=k,
        scheme=scheme,
        generate_fn=qwen_generate_fn,
        rerank=rerank,
        candidate_k=candidate_k,
    )

    retrieved_ids = [c["id"] for c in chunks if "id" in c]

    return {
        "answer": reply_text,
        "latency": round(latency, 4),
        "retrieved_ids": retrieved_ids,
    }
