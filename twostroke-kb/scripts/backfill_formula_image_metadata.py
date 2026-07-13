"""Repair existing DB rows after formula/image features were added.

Run from the project root:
    python scripts/backfill_formula_image_metadata.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_connection, get_settings
from ingestion.formula_extractor import extract_formulas
from ingestion.knowledge_base import ensure_document_images_table, ensure_formulas_table
from ingestion.parsers.math_normalizer import normalize_math_unicode


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_.-]", "_", value.lower()).strip("_")


def _doc_stems(cur: Any) -> list[dict[str, str]]:
    cur.execute("SELECT doc_id, min(metadata->>'filename') FROM chunks GROUP BY doc_id")
    docs = []
    for doc_id, filename in cur.fetchall():
        name = filename or doc_id
        docs.append({
            "doc_id": doc_id,
            "filename": name,
            "stem": _slug(Path(name).stem),
            "doc_stem": _slug(Path(doc_id).stem),
        })
    return docs


def _match_doc(prefix: str, docs: list[dict[str, str]]) -> str | None:
    prefix = _slug(re.sub(r"__\d+__$", "", prefix))
    best: tuple[int, str] | None = None
    for doc in docs:
        candidates = {doc["stem"], doc["doc_stem"]}
        for cand in candidates:
            cand = _slug(re.sub(r"__\d+__$", "", cand))
            if not cand:
                continue
            if prefix.startswith(cand) or cand.startswith(prefix):
                score = min(len(prefix), len(cand))
                if best is None or score > best[0]:
                    best = (score, doc["doc_id"])
    return best[1] if best else None


def _nearest_chunk_id(cur: Any, doc_id: str, page_or_slide: int | None) -> int | None:
    if page_or_slide is not None:
        cur.execute(
            """
            SELECT id FROM chunks
            WHERE doc_id = %s
              AND ((metadata->>'page')::int = %s OR (metadata->>'slide')::int = %s)
            ORDER BY id
            LIMIT 1
            """,
            (doc_id, page_or_slide, page_or_slide),
        )
        row = cur.fetchone()
        if row:
            return row[0]
    cur.execute("SELECT id FROM chunks WHERE doc_id = %s ORDER BY id LIMIT 1", (doc_id,))
    row = cur.fetchone()
    return row[0] if row else None


def _normalize_chunks(cur: Any) -> int:
    cur.execute("SELECT id, content FROM chunks")
    updated = 0
    for chunk_id, content in cur.fetchall():
        normalized = normalize_math_unicode(content or "")
        if normalized != (content or ""):
            cur.execute("UPDATE chunks SET content = %s WHERE id = %s", (normalized, chunk_id))
            updated += 1
    return updated


def _backfill_formulas(cur: Any) -> int:
    ensure_formulas_table(cur)
    cur.execute("DELETE FROM formulas")
    cur.execute("SELECT id, doc_id, content, metadata, source_refs FROM chunks ORDER BY id")
    inserted = 0
    for chunk_id, doc_id, content, metadata, source_refs in cur.fetchall():
        metadata = metadata or {}
        refs = source_refs or [{}]
        ref = refs[0] if refs else {}
        page_or_slide = metadata.get("page") or metadata.get("slide") or ref.get("page") or ref.get("slide")
        topic = metadata.get("topic") or ref.get("topic")
        source_title = metadata.get("source_title") or ref.get("source_title") or ref.get("filename") or metadata.get("filename")
        for formula in extract_formulas(content or ""):
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
            inserted += 1
    return inserted


def _backfill_images(cur: Any) -> int:
    ensure_document_images_table(cur)
    docs = _doc_stems(cur)
    image_dir = Path(get_settings().image_store_path)
    if not image_dir.exists():
        return 0

    inserted = 0
    pattern = re.compile(r"^(?P<prefix>.+)_(?P<kind>page|slide)(?P<num>\d+)_img(?P<idx>\d+)\.(?P<ext>png|jpe?g|webp)$", re.I)
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        doc_id = _match_doc(match.group("prefix"), docs)
        if not doc_id:
            continue
        page_or_slide = int(match.group("num"))
        image_index = int(match.group("idx"))
        url = f"/images/{path.name}"
        cur.execute(
            """
            SELECT 1 FROM document_images
            WHERE doc_id = %s AND page_or_slide = %s AND image_index = %s AND url = %s
            LIMIT 1
            """,
            (doc_id, page_or_slide, image_index, url),
        )
        if cur.fetchone():
            continue
        chunk_id = _nearest_chunk_id(cur, doc_id, page_or_slide)
        cur.execute(
            """
            INSERT INTO document_images
                (doc_id, page_or_slide, image_index, file_path, url, caption, chunk_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (doc_id, page_or_slide, image_index, str(path), url, "", chunk_id),
        )
        inserted += 1
    return inserted


def main() -> None:
    conn = get_connection()
    try:
        with conn.transaction():
            cur = conn.cursor()
            normalized = _normalize_chunks(cur)
            formulas = _backfill_formulas(cur)
            images = _backfill_images(cur)
        print(f"Normalized chunks: {normalized}")
        print(f"Formula rows inserted: {formulas}")
        print(f"Image metadata rows inserted: {images}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
