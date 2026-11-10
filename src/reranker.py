"""Cross-encoder reranking for the farmer-scheme retriever.

The embedding index is intentionally kept as the first-stage retriever.  A
cross-encoder only scores a small candidate set, which is more accurate than
asking the embedding model to order every chunk and is still inexpensive on a
Colab CPU/T4.
"""

from __future__ import annotations

from typing import Any, Dict, List

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_CACHE: dict[tuple[str, str], Any] = {}


def get_reranker(model_name: str = DEFAULT_RERANKER_MODEL, device: str = "cpu") -> Any:
    """Load and cache a sentence-transformers CrossEncoder lazily."""
    key = (model_name, device)
    if key not in _CACHE:
        from sentence_transformers import CrossEncoder

        _CACHE[key] = CrossEncoder(model_name, device=device)
    return _CACHE[key]


def rerank(
    question: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
    model_name: str = DEFAULT_RERANKER_MODEL,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Rerank candidate chunks and return copies with both scores preserved."""
    if not candidates:
        return []
    model = get_reranker(model_name=model_name, device=device)
    pairs = [(question, str(chunk.get("text", ""))) for chunk in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = []
    for chunk, score in zip(candidates, scores):
        item = dict(chunk)
        item["embedding_score"] = item.get("score")
        item["rerank_score"] = round(float(score), 6)
        item["score"] = item["rerank_score"]
        ranked.append(item)
    ranked.sort(key=lambda row: row["rerank_score"], reverse=True)
    return ranked[:top_k]
