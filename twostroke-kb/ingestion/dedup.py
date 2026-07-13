"""Deduplicate near-identical chunks at ingest.

RULE (see CLAUDE.md): when cosine > threshold, KEEP one chunk but MERGE the
new source reference onto it. NEVER drop a source — conflict_check + citations
need to know every place a fact appears.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from config import get_settings

log = logging.getLogger(__name__)


def dedup_and_merge(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare each chunk's embedding against already-stored vectors AND prior
    chunks in the same batch.

    If cosine similarity > DEDUP_COSINE_THRESHOLD (default 0.98):
      - Merge the new chunk's source_refs onto the existing chunk (DB or batch).
      - Drop the duplicate from the return list.

    Returns the subset of chunks that should actually be inserted.
    """
    import numpy as np

    threshold = get_settings().dedup_cosine_threshold
    kept: list[dict[str, Any]] = []
    # batch embeddings for within-batch comparison (list of (index, vector))
    batch_vecs: list[tuple[int, list[float]]] = []

    for chunk in chunks:
        embedding = chunk.get("embedding")
        if not embedding:
            kept.append(chunk)
            continue

        # 1. Check within-batch first (no DB round-trip needed)
        batch_dup_idx = _find_batch_duplicate(embedding, batch_vecs, threshold)
        if batch_dup_idx is not None:
            log.info("dedup: batch duplicate; merging provenance onto batch index %d", batch_dup_idx)
            existing = kept[batch_dup_idx]
            existing["source_refs"] = _merge_refs_unique(
                existing.get("source_refs", []), chunk.get("source_refs", [])
            )
            continue

        # 2. Check against DB
        existing_id = _find_near_duplicate(embedding, threshold)
        if existing_id is not None:
            log.info("dedup: DB duplicate (cosine > %.2f); merging provenance onto id=%d", threshold, existing_id)
            _merge_source_refs(existing_id, chunk.get("source_refs", []))
            continue

        batch_vecs.append((len(kept), embedding))
        kept.append(chunk)

    return kept


def _find_batch_duplicate(
    embedding: list[float],
    batch_vecs: list[tuple[int, list[float]]],
    threshold: float,
) -> int | None:
    """Return the kept-list index of the first batch vector above threshold, else None."""
    import numpy as np

    if not batch_vecs:
        return None

    vec = np.array(embedding, dtype="float32")
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    vec = vec / norm

    for kept_idx, bvec in batch_vecs:
        bv = np.array(bvec, dtype="float32")
        bn = np.linalg.norm(bv)
        if bn == 0:
            continue
        cosine = float(np.dot(vec, bv / bn))
        if cosine >= threshold:
            return kept_idx
    return None


def _find_near_duplicate(embedding: list[float], threshold: float) -> int | None:
    """Return the id of the nearest stored chunk if cosine > threshold, else None."""
    from config import get_connection

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, 1 - (embedding <=> %s::vector) AS cosine
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT 1
            """,
            (str(embedding), str(embedding)),
        )
        row = cur.fetchone()
        if row and float(row[1]) >= threshold:
            return int(row[0])
        return None
    finally:
        conn.close()


def _merge_refs_unique(
    existing: list[dict[str, Any]], new_refs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Append new_refs to existing, skipping exact duplicates (by JSON equality)."""
    seen = {json.dumps(r, sort_keys=True) for r in existing}
    result = list(existing)
    for ref in new_refs:
        key = json.dumps(ref, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(ref)
    return result


def _merge_source_refs(chunk_id: int, new_refs: list[dict[str, Any]]) -> None:
    """Append new_refs into the existing chunk's source_refs JSONB column, no duplicates.

    Reads current refs first, merges in Python, then writes back atomically.
    """
    if not new_refs:
        return

    from config import get_connection

    conn = get_connection()
    try:
        with conn.transaction():
            cur = conn.cursor()
            cur.execute("SELECT source_refs FROM chunks WHERE id = %s", (chunk_id,))
            row = cur.fetchone()
            if not row:
                return
            current = row[0] if isinstance(row[0], list) else json.loads(row[0] or "[]")
            merged = _merge_refs_unique(current, new_refs)
            cur.execute(
                "UPDATE chunks SET source_refs = %s::jsonb WHERE id = %s",
                (json.dumps(merged), chunk_id),
            )
    finally:
        conn.close()
