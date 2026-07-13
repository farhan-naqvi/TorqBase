"""Memory layers: conversation (session), user profile (long-term), feedback.

NOTE: none of this retrains the model; it changes context and retrieval weights.
"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)


def get_conversation(session_id: str) -> list[dict[str, Any]]:
    """Return prior turns for follow-up questions. Returns [] if session is new."""
    from config import get_connection

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT turns FROM conversations WHERE session_id = %s",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return []
        turns = row[0]
        return json.loads(turns) if isinstance(turns, str) else (turns or [])
    finally:
        conn.close()


def append_turn(session_id: str, role: str, content: str) -> None:
    """Append one turn to the conversation; create the session row if absent."""
    from config import get_connection

    turn = json.dumps([{"role": role, "content": content}])
    conn = get_connection()
    try:
        with conn.transaction():
            cur = conn.cursor()
            _ensure_chunk_quality_columns(cur)
            cur.execute(
                """
                INSERT INTO conversations (session_id, turns)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (session_id) DO UPDATE
                SET turns      = conversations.turns || %s::jsonb,
                    updated_at = now()
                """,
                (session_id, turn, turn),
            )
    finally:
        conn.close()


def get_profile(user_id: str) -> dict[str, Any]:
    """Return the user profile, or sensible defaults for a new user."""
    from config import get_connection

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT lang_pref, expertise, engines, props FROM user_profile WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return {"lang_pref": None, "expertise": "expert", "engines": [], "props": {}}

    lang_pref, expertise, engines, props = row
    return {
        "lang_pref": lang_pref,
        "expertise": expertise or "expert",
        "engines": list(engines) if engines else [],
        "props": json.loads(props) if isinstance(props, str) else (props or {}),
    }


def record_feedback(
    session_id: str,
    question: str,
    answer: str,
    vote: int = 0,
    correction: str = "",
    expert_note: str = "",
    chunk_ids: list[int] | None = None,
) -> None:
    """Store feedback, adjust cited chunk quality, and promote expert corrections."""
    from config import get_connection

    conn = get_connection()
    try:
        with conn.transaction():
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO feedback
                    (session_id, question, answer, vote, correction, expert_note, chunk_ids)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    question,
                    answer,
                    vote,
                    correction or "",
                    expert_note or "",
                    chunk_ids or [],
                ),
            )

            if chunk_ids and vote in (1, -1):
                for chunk_id in chunk_ids:
                    if vote == 1:
                        cur.execute(
                            """
                            UPDATE chunks
                            SET vote_count = vote_count + 1,
                                quality_score = (
                                    (vote_count::float * quality_score + 1.0)
                                    / (vote_count + 1)
                                )
                            WHERE id = %s
                            """,
                            (chunk_id,),
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE chunks
                            SET vote_count = vote_count + 1,
                                downvote_count = downvote_count + 1,
                                quality_score = (
                                    (vote_count::float * quality_score - 1.0)
                                    / (vote_count + 1)
                                )
                            WHERE id = %s
                            """,
                            (chunk_id,),
                        )

            if correction and len(correction.strip()) > 20:
                try:
                    _store_correction_as_chunk(question, correction.strip(), session_id, cur)
                except Exception:
                    log.exception("store_feedback: failed to store correction as chunk")
    finally:
        conn.close()

    try:
        from agent.retriever_hybrid import invalidate_bm25_cache

        invalidate_bm25_cache()
    except Exception:
        pass


def _store_correction_as_chunk(question: str, correction: str, session_id: str, cur: Any) -> None:
    """Store an expert correction as high-trust searchable knowledge."""
    from ingestion.knowledge_base import _embedder

    content = (
        f"Expert correction for question: {question}\n\n"
        f"Verified answer: {correction}\n\n"
        f"Source: Expert correction submitted by engineer (session: {session_id})"
    )
    raw_embedding = _embedder().encode([content], show_progress_bar=False)[0]
    embedding = raw_embedding.tolist() if hasattr(raw_embedding, "tolist") else list(raw_embedding)
    metadata = {
        "source_type": "expert_correction",
        "filename": "Expert Correction",
        "source_title": "Expert Correction",
        "original_question": question,
        "topic": "Expert Corrections",
        "chunk_type": "correction",
        "verified": True,
    }
    source_refs = [{"source": "expert_correction", "session_id": session_id}]
    cur.execute(
        """
        INSERT INTO chunks
            (doc_id, content, lang, embedding, metadata, source_refs, quality_score, vote_count)
        VALUES
            ('expert-corrections', %s, 'unknown', %s::vector, %s::jsonb, %s::jsonb, 1.0, 1)
        """,
        (content, str(embedding), json.dumps(metadata), json.dumps(source_refs)),
    )
    log.info("store_feedback: correction stored as high-trust chunk")


def _ensure_chunk_quality_columns(cur: Any) -> None:
    cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS quality_score FLOAT DEFAULT 0.0")
    cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS vote_count INT DEFAULT 0")
    cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS downvote_count INT DEFAULT 0")
