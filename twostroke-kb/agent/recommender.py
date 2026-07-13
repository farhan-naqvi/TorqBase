"""Recommendation (a challenge requirement): related questions + graph neighbours.

Two-stage strategy:
  1. Graph neighbours — query graph_nodes/graph_edges for entities from the question.
     Returns empty list gracefully when the graph is unpopulated (graph_builder is Slice 5).
  2. LLM generation — ask the model for follow-ups given the retrieved context snippets.

Caller gets up to 3 deduplicated strings.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def related(question: str, used_chunks: list[dict[str, Any]]) -> list[str]:
    """Return up to 3 follow-up questions a technician might ask next.

    Tries graph neighbours first (fast, citation-free), fills remaining slots
    with LLM-generated questions from the retrieved context.
    """
    suggestions: list[str] = _graph_neighbours(question)
    remaining = 3 - len(suggestions)
    if remaining > 0:
        suggestions += _llm_suggestions(question, used_chunks, n=remaining)
    return suggestions[:3]


def _graph_neighbours(question: str) -> list[str]:
    """Find entity names related to words in the question via graph_edges.

    Returns [] when the graph is empty or the DB is unavailable.
    """
    try:
        from config import get_connection

        words = [w for w in question.split() if len(w) > 3][:6]
        if not words:
            return []

        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT DISTINCT n2.name, n2.type
                FROM graph_nodes n1
                JOIN graph_edges e  ON e.src_id = n1.id
                JOIN graph_nodes n2 ON n2.id    = e.dst_id
                WHERE n1.name ILIKE ANY(%s)
                LIMIT 5
                """,
                ([f"%{w}%" for w in words],),
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        return [f"Tell me more about {row[0]}." for row in rows]
    except Exception:
        log.debug("recommender: graph lookup unavailable")
        return []


def _llm_suggestions(
    question: str, chunks: list[dict[str, Any]], n: int = 3
) -> list[str]:
    """Ask the LLM to generate n follow-up questions from the retrieved snippets."""
    if not chunks or n <= 0:
        return []
    try:
        from llm import chat_json

        snippet = "\n".join(
            str(c.get("content") or c.get("value") or "")[:200]
            for c in chunks[:3]
        )
        raw = chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        f"Suggest {n} concise follow-up questions a two-stroke engine "
                        "technician might ask after this query, based on the context. "
                        "Return a JSON array of strings only."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContext:\n{snippet}",
                },
            ],
            max_tokens=150,
        )
        if isinstance(raw, list):
            return [str(q) for q in raw[:n] if q]
        return []
    except Exception:
        log.debug("recommender: LLM suggestion failed")
        return []
