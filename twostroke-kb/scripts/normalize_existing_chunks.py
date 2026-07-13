"""One-time cleanup for already-ingested math Unicode in stored chunks/formulas.

Run from the project root:
    python scripts/normalize_existing_chunks.py
"""
from __future__ import annotations

from config import get_connection
from ingestion.parsers.math_normalizer import normalize_math_unicode


def _normalize_chunks(cur) -> int:
    cur.execute("SELECT id, content FROM chunks")
    rows = cur.fetchall()
    updated = 0
    for chunk_id, content in rows:
        normalized = normalize_math_unicode(content or "")
        if normalized != (content or ""):
            cur.execute("UPDATE chunks SET content = %s WHERE id = %s", (normalized, chunk_id))
            updated += 1
    return updated


def _normalize_formulas(cur) -> int:
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'formulas'
        )
        """
    )
    if not cur.fetchone()[0]:
        return 0

    cur.execute("SELECT id, formula_text, formula_latex, context_before, context_after FROM formulas")
    rows = cur.fetchall()
    updated = 0
    for row in rows:
        formula_id, formula_text, formula_latex, context_before, context_after = row
        values = {
            "formula_text": normalize_math_unicode(formula_text or ""),
            "formula_latex": normalize_math_unicode(formula_latex or "") if formula_latex else None,
            "context_before": normalize_math_unicode(context_before or ""),
            "context_after": normalize_math_unicode(context_after or ""),
        }
        if (
            values["formula_text"] != (formula_text or "")
            or values["formula_latex"] != formula_latex
            or values["context_before"] != (context_before or "")
            or values["context_after"] != (context_after or "")
        ):
            cur.execute(
                """
                UPDATE formulas
                SET formula_text = %s, formula_latex = %s,
                    context_before = %s, context_after = %s
                WHERE id = %s
                """,
                (
                    values["formula_text"],
                    values["formula_latex"],
                    values["context_before"],
                    values["context_after"],
                    formula_id,
                ),
            )
            updated += 1
    return updated


def main() -> None:
    conn = get_connection()
    try:
        with conn.transaction():
            cur = conn.cursor()
            chunk_count = _normalize_chunks(cur)
            formula_count = _normalize_formulas(cur)
        print(f"Normalized {chunk_count} chunks and {formula_count} formula rows.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
