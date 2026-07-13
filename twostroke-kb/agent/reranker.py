"""Cross-encoder re-ranker: rescore top-20 candidates, keep top-5 for the LLM."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from config import get_settings


@lru_cache
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model)


def rerank(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score every (query, chunk) pair with a cross-encoder and return the top RERANK_TOP_K.

    Attaches 'rerank_score' to each returned chunk.
    Returns candidates unchanged (sorted by original score) if list is empty.
    """
    if not candidates:
        return candidates

    model = _model()
    pairs = [(query, c["content"]) for c in candidates]
    scores = model.predict(pairs)

    top_k = get_settings().rerank_top_k
    ranked = sorted(zip(scores, candidates), key=lambda x: float(x[0]), reverse=True)[:top_k]

    result = []
    for score, chunk in ranked:
        chunk = dict(chunk)
        chunk["rerank_score"] = float(score)
        result.append(chunk)

    return result
