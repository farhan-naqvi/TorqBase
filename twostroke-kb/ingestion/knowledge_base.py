"""Embed chunks and persist to pgvector; persist structured_facts from tables."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import get_settings
from .types import ParsedDoc


@lru_cache
def _embedder():
    """Load the multilingual embedding model once."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def embed(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Batch-encode chunk content and attach an 'embedding' list to each chunk.

    Raises ValueError if the model returns unexpected dimension.
    """
    if not chunks:
        return chunks

    settings = get_settings()
    model = _embedder()
    texts = [c["content"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=False)

    for c, vec in zip(chunks, vectors):
        if len(vec) != settings.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: got {len(vec)}, expected {settings.embedding_dim}"
            )
        c["embedding"] = vec.tolist()

    return chunks


def _doc_id_slug(filename: str) -> str:
    """Lowercase filename, replace spaces and special chars with underscores."""
    return re.sub(r"[^a-z0-9_.]", "_", filename.lower())


def store(chunks: list[dict[str, Any]]) -> None:
    """Insert chunks into the `chunks` table and table rows into `structured_facts`.

    Each chunk must already have an 'embedding' key (call embed() first).
    Uses a single transaction per batch.
    """
    from config import get_connection

    if not chunks:
        return

    conn = get_connection()
    try:
        with conn.transaction():
            cur = conn.cursor()
            for c in chunks:
                doc_id = _doc_id_slug(c["metadata"].get("filename", "unknown"))
                lang = c["metadata"].get("lang", "unknown")
                embedding = c["embedding"]
                metadata = {k: v for k, v in c["metadata"].items() if k != "embedding"}
                source_refs = json.dumps(c.get("source_refs", []))

                cur.execute(
                    """
                    INSERT INTO chunks (doc_id, content, lang, embedding, metadata, source_refs)
                    VALUES (%s, %s, %s, %s::vector, %s::jsonb, %s::jsonb)
                    RETURNING id
                    """,
                    (
                        doc_id,
                        c["content"],
                        lang,
                        str(embedding),
                        json.dumps(metadata),
                        source_refs,
                    ),
                )
                chunk_id = int(cur.fetchone()[0])
                _store_chunk_formulas(cur, doc_id, chunk_id, c)

                # Write table cells to structured_facts for exact spec lookup
                if c["metadata"].get("chunk_type") == "table":
                    _store_table_facts(cur, doc_id, c)

        try:
            from agent.retriever_hybrid import invalidate_bm25_cache

            invalidate_bm25_cache()
        except Exception:
            pass

    finally:
        conn.close()


def store_document_images(doc: ParsedDoc) -> None:
    """Persist extracted document image metadata and link to nearby page/slide chunks."""
    from config import get_connection

    if not doc.images:
        return

    filename = doc.metadata.get("filename", "unknown")
    doc_id = _doc_id_slug(filename)

    conn = get_connection()
    try:
        with conn.transaction():
            cur = conn.cursor()
            ensure_document_images_table(cur)
            cur.execute("SELECT 1 FROM document_images WHERE doc_id = %s LIMIT 1", (doc_id,))
            if cur.fetchone() is not None:
                return

            for image in doc.images:
                page_or_slide = image.get("page") or image.get("slide")
                chunk_id = _nearest_chunk_id(cur, doc_id, page_or_slide, has_slide="slide" in image)
                filename = image.get("filename") or Path(str(image.get("path", ""))).name
                url = image.get("url") or (f"/images/{filename}" if filename else "")
                cur.execute(
                    """
                    INSERT INTO document_images
                        (doc_id, page_or_slide, image_index, file_path, url, caption, chunk_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        doc_id,
                        page_or_slide,
                        image.get("index"),
                        str(image.get("path", "")),
                        url,
                        image.get("caption", ""),
                        chunk_id,
                    ),
                )
    finally:
        conn.close()


def store_images(doc_id: str, images: list[dict]) -> None:
    """Compatibility wrapper for image metadata already extracted by parsers."""
    if not images:
        return
    doc = ParsedDoc(text="", metadata={"filename": doc_id}, source_ref={}, images=images)
    store_document_images(doc)


def ensure_document_images_table(cur: Any) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS document_images (
            id            SERIAL PRIMARY KEY,
            doc_id        TEXT NOT NULL,
            page_or_slide INT,
            image_index   INT,
            file_path     TEXT NOT NULL,
            url           TEXT NOT NULL DEFAULT '',
            caption       TEXT DEFAULT '',
            chunk_id      BIGINT REFERENCES chunks(id),
            created_at    TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    cur.execute("ALTER TABLE document_images ADD COLUMN IF NOT EXISTS url TEXT NOT NULL DEFAULT ''")
    cur.execute("CREATE INDEX IF NOT EXISTS document_images_chunk_idx ON document_images (chunk_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS document_images_doc_location_idx ON document_images (doc_id, page_or_slide)")


def ensure_formulas_table(cur: Any) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS formulas (
            id             SERIAL PRIMARY KEY,
            doc_id         TEXT NOT NULL,
            chunk_id       BIGINT REFERENCES chunks(id),
            formula_text   TEXT NOT NULL,
            formula_latex  TEXT,
            context_before TEXT,
            context_after  TEXT,
            page_or_slide  INT,
            topic          TEXT,
            source_title   TEXT,
            created_at     TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS formulas_doc_idx ON formulas (doc_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS formulas_topic_idx ON formulas (topic)")


def _store_chunk_formulas(cur: Any, doc_id: str, chunk_id: int, chunk: dict[str, Any]) -> None:
    from ingestion.formula_extractor import extract_formulas

    formulas = extract_formulas(chunk.get("content", ""))
    if not formulas:
        return

    ensure_formulas_table(cur)
    metadata = chunk.get("metadata") or {}
    refs = chunk.get("source_refs") or [{}]
    ref = refs[0] if refs else {}
    page_or_slide = metadata.get("page") or metadata.get("slide") or ref.get("page") or ref.get("slide")
    topic = metadata.get("topic") or ref.get("topic")
    source_title = metadata.get("source_title") or ref.get("source_title") or ref.get("filename") or metadata.get("filename")

    for formula in formulas:
        cur.execute(
            """
            INSERT INTO formulas
                (doc_id, chunk_id, formula_text, formula_latex, context_before,
                 context_after, page_or_slide, topic, source_title)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                doc_id,
                chunk_id,
                formula["formula_text"],
                formula.get("formula_latex"),
                formula.get("context_before", ""),
                formula.get("context_after", ""),
                page_or_slide,
                topic,
                source_title,
            ),
        )


def _nearest_chunk_id(cur: Any, doc_id: str, page_or_slide: Any, has_slide: bool) -> int | None:
    if page_or_slide is None:
        return None

    key = "slide" if has_slide else "page"
    cur.execute(
        f"""
        SELECT id
        FROM chunks
        WHERE doc_id = %s
          AND metadata->>%s = %s
        ORDER BY COALESCE((metadata->>'chunk_index')::int, 0)
        LIMIT 1
        """,
        (doc_id, key, str(page_or_slide)),
    )
    row = cur.fetchone()
    if row:
        return int(row[0])

    cur.execute(
        "SELECT id FROM chunks WHERE doc_id = %s ORDER BY id LIMIT 1",
        (doc_id,),
    )
    row = cur.fetchone()
    return int(row[0]) if row else None


def _store_table_facts(cur: Any, doc_id: str, chunk: dict[str, Any]) -> None:
    """Parse a table chunk and insert individual cell values into structured_facts."""
    lines = chunk["content"].splitlines()
    if not lines:
        return

    # First data row is the header (after optional [Table: name] line)
    header: list[str] = []
    data_rows: list[list[str]] = []
    units_map: dict[str, str] = {}  # col_label -> unit
    for line in lines:
        if line.startswith("[Table:"):
            continue
        if line.startswith("Units:"):
            # format: "Units: col1=unit1, col2=unit2"
            for part in line[len("Units:"):].strip().split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    units_map[k.strip()] = v.strip()
            continue
        cells = [c.strip() for c in line.split("|")]
        if not header:
            header = cells
        else:
            data_rows.append(cells)

    sheet_name = chunk["metadata"].get("table_name", "")
    source_ref = json.dumps(chunk.get("source_refs", [{}])[0])

    for row in data_rows:
        row_label = row[0] if row else ""
        for col_idx, value in enumerate(row[1:], start=1):
            col_label = header[col_idx] if col_idx < len(header) else ""
            if value.strip():
                unit = units_map.get(col_label)
                cur.execute(
                    """
                    INSERT INTO structured_facts
                        (doc_id, sheet, row_label, col_label, key, value, unit, source_ref)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        doc_id,
                        sheet_name,
                        row_label,
                        col_label,
                        f"{row_label}::{col_label}",
                        value.strip(),
                        unit,
                        source_ref,
                    ),
                )
