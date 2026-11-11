"""Hybrid lexical + semantic retrieval with reranking and abstention."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.rag import DEFAULT_EMBEDDING_MODEL, get_embedding_model, load_index
from src.reranker import DEFAULT_RERANKER_MODEL, rerank as cross_encoder_rerank


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


class HybridRetriever:
    """BM25 + BGE reciprocal-rank fusion, followed by a cross-encoder."""

    def __init__(self, candidate_k: int = 20, top_k: int = 4,
                 reranker_model: str = DEFAULT_RERANKER_MODEL,
                 min_rerank_score: Optional[float] = None):
        from rank_bm25 import BM25Okapi

        self.candidate_k = max(top_k, candidate_k)
        self.top_k = top_k
        self.reranker_model = reranker_model
        self.min_rerank_score = min_rerank_score
        self.index, self.chunks = load_index()
        self.embedder = get_embedding_model(DEFAULT_EMBEDDING_MODEL, device="cpu")
        self.bm25 = BM25Okapi([_tokens(c.get("text", "")) for c in self.chunks])

    def __call__(self, question: str, k: int = 4) -> List[Dict[str, Any]]:
        from src.rag import retrieve

        semantic = retrieve(question, k=self.candidate_k, index=self.index,
                             chunks=self.chunks, rerank=False)
        semantic_rank = {row["id"]: rank for rank, row in enumerate(semantic, 1)}
        lexical_scores = self.bm25.get_scores(_tokens(question))
        lexical_ids = sorted(range(len(lexical_scores)), key=lambda i: lexical_scores[i], reverse=True)[:self.candidate_k]
        lexical_rank = {self.chunks[i]["id"]: rank for rank, i in enumerate(lexical_ids, 1)}

        by_id = {row["id"]: row for row in semantic}
        for i in lexical_ids:
            by_id.setdefault(self.chunks[i]["id"], dict(self.chunks[i]))
        fused = []
        for cid, row in by_id.items():
            score = (1.0 / (60 + semantic_rank[cid]) if cid in semantic_rank else 0.0)
            score += (1.0 / (60 + lexical_rank[cid]) if cid in lexical_rank else 0.0)
            item = dict(row)
            item["rrf_score"] = score
            fused.append(item)
        fused.sort(key=lambda row: row["rrf_score"], reverse=True)
        candidates = fused[:self.candidate_k]
        ranked = cross_encoder_rerank(question, candidates, k, self.reranker_model, "cpu")
        if self.min_rerank_score is not None and (not ranked or ranked[0].get("rerank_score", -1e9) < self.min_rerank_score):
            return []
        return ranked
