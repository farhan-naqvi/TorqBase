"""Tools the agent can call (model-driven function calling).

Each tool returns evidence with source references attached.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any


def hybrid_search(query: str, k: int = 20) -> list[dict[str, Any]]:
    """BM25 + dense retrieval -> top-k candidate chunks (pre-rerank)."""
    from .retriever_hybrid import search

    return search(query, k=k)


def graph_lookup(entity: str, relation: str | None = None) -> list[dict[str, Any]]:
    """Traverse Engine->Part->Symptom->Cause->Fix from graph tables.

    Returns edges whose source node name matches entity (ILIKE).
    Optionally filter by relation type.
    """
    from config import get_connection

    conn = get_connection()
    try:
        cur = conn.cursor()
        if relation:
            cur.execute(
                """
                SELECT n1.type, n1.name, e.relation, n2.type, n2.name
                FROM graph_nodes n1
                JOIN graph_edges e  ON e.src_id = n1.id
                JOIN graph_nodes n2 ON n2.id    = e.dst_id
                WHERE n1.name ILIKE %s AND e.relation ILIKE %s
                LIMIT 20
                """,
                (f"%{entity}%", f"%{relation}%"),
            )
        else:
            cur.execute(
                """
                SELECT n1.type, n1.name, e.relation, n2.type, n2.name
                FROM graph_nodes n1
                JOIN graph_edges e  ON e.src_id = n1.id
                JOIN graph_nodes n2 ON n2.id    = e.dst_id
                WHERE n1.name ILIKE %s
                LIMIT 20
                """,
                (f"%{entity}%",),
            )
        rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "content": f"{r[1]} ({r[0]}) --[{r[2]}]--> {r[4]} ({r[3]})",
            "src_type": r[0], "src_name": r[1],
            "relation": r[2],
            "dst_type": r[3], "dst_name": r[4],
        }
        for r in rows
    ]


def spec_lookup(key: str, engine: str | None = None) -> list[dict[str, Any]]:
    """Return EXACT values from structured_facts (never computed). Always cite.

    Matches key with ILIKE (case-insensitive substring). If engine is provided,
    also filters by row_label or doc_id containing that string.
    Returns at most 20 rows, each with source_ref for citation.
    """
    from config import get_connection

    conn = get_connection()
    try:
        cur = conn.cursor()
        if engine:
            cur.execute(
                """
                SELECT doc_id, sheet, row_label, col_label, key, value, unit, source_ref
                FROM structured_facts
                WHERE key ILIKE %s
                  AND (row_label ILIKE %s OR doc_id ILIKE %s)
                LIMIT 20
                """,
                (f"%{key}%", f"%{engine}%", f"%{engine}%"),
            )
        else:
            cur.execute(
                """
                SELECT doc_id, sheet, row_label, col_label, key, value, unit, source_ref
                FROM structured_facts
                WHERE key ILIKE %s
                LIMIT 20
                """,
                (f"%{key}%",),
            )
        rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "doc_id": r[0],
            "sheet": r[1],
            "row_label": r[2],
            "col_label": r[3],
            "key": r[4],
            "value": r[5],
            "unit": r[6],
            "source_ref": r[7],
            "content": f"{r[4]} = {r[5]}{' ' + r[6] if r[6] else ''} (source: {r[0]})",
        }
        for r in rows
    ]


def conflict_check(claim: str) -> list[dict[str, Any]]:
    """Find structured_facts rows where the same key has conflicting values across sources.

    Returns all matching rows tagged with conflict=True when multiple distinct
    values exist for the same key — so the agent can surface the disagreement.
    """
    from config import get_connection

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT key, value, doc_id, source_ref
            FROM structured_facts
            WHERE key ILIKE %s
            ORDER BY key, value
            LIMIT 40
            """,
            (f"%{claim}%",),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, value, doc_id, source_ref in rows:
        by_key[key].append({"value": value, "doc_id": doc_id, "source_ref": source_ref})

    result: list[dict[str, Any]] = []
    for key, facts in by_key.items():
        values = {f["value"] for f in facts}
        has_conflict = len(values) > 1
        for f in facts:
            result.append({
                "content": f"{key} = {f['value']} (from {f['doc_id']})"
                           + (" ⚠ CONFLICT" if has_conflict else ""),
                "key": key,
                "value": f["value"],
                "doc_id": f["doc_id"],
                "source_ref": f["source_ref"],
                "conflict": has_conflict,
            })

    return result


# Conversion table: (normalised_from, normalised_to) -> multiplier
_CONV: dict[tuple[str, str], float] = {
    ("nm", "lb-ft"): 0.73756,
    ("lb-ft", "nm"): 1.35582,
    ("bar", "psi"): 14.5038,
    ("psi", "bar"): 0.068948,
    ("kw", "hp"): 1.34102,
    ("hp", "kw"): 0.74570,
    ("ps", "kw"): 0.73550,
    ("kw", "ps"): 1.35962,
    ("mm", "in"): 0.039370,
    ("in", "mm"): 25.4,
    ("ml", "l"): 0.001,
    ("l", "ml"): 1000.0,
    ("kg", "lb"): 2.20462,
    ("lb", "kg"): 0.45359,
    ("rpm", "rps"): 1 / 60,
    ("rps", "rpm"): 60.0,
    ("rpm", "rad/s"): math.pi / 30,
    ("rad/s", "rpm"): 30 / math.pi,
}

_TEMP_ALIASES = {
    "°c": "c", "degc": "c", "celsius": "c",
    "°f": "f", "degf": "f", "fahrenheit": "f",
    "k": "k", "kelvin": "k",
}


def unit_convert(value: float, from_unit: str, to_unit: str) -> float:
    """Deterministic unit conversion from a fixed lookup table — NOT via the LLM."""
    fu = from_unit.lower().strip()
    tu = to_unit.lower().strip()

    if fu == tu:
        return value

    # Temperature special cases
    fa = _TEMP_ALIASES.get(fu)
    ta = _TEMP_ALIASES.get(tu)
    if fa and ta:
        if fa == ta:
            return value
        if fa == "c" and ta == "f":
            return value * 9 / 5 + 32
        if fa == "f" and ta == "c":
            return (value - 32) * 5 / 9
        if fa == "c" and ta == "k":
            return value + 273.15
        if fa == "k" and ta == "c":
            return value - 273.15
        if fa == "f" and ta == "k":
            return (value - 32) * 5 / 9 + 273.15
        if fa == "k" and ta == "f":
            return (value - 273.15) * 9 / 5 + 32

    key = (fu, tu)
    if key not in _CONV:
        raise ValueError(f"unit_convert: no conversion from {from_unit!r} to {to_unit!r}")
    return value * _CONV[key]


def diagnostic_tree(symptom: str, engine: str | None = None) -> list[dict[str, Any]]:
    """Walk graph: symptom -> cause -> fix. Returns structured diagnostic paths."""
    from config import get_connection

    conn = get_connection()
    try:
        cur = conn.cursor()
        params: list[Any] = [f"%{symptom}%"]
        engine_join = ""
        if engine:
            engine_join = (
                "JOIN graph_edges e0 ON e0.dst_id = n_sym.id "
                "JOIN graph_nodes n_eng ON n_eng.id = e0.src_id "
                "  AND n_eng.type = 'engine' AND n_eng.name ILIKE %s "
            )
            params.insert(0, f"%{engine}%")

        cur.execute(
            f"""
            SELECT n_sym.name, e1.relation, n_cause.name, e2.relation, n_fix.name
            FROM graph_nodes n_sym
            {engine_join}
            JOIN graph_edges e1    ON e1.src_id   = n_sym.id
            JOIN graph_nodes n_cause ON n_cause.id = e1.dst_id AND n_cause.type = 'cause'
            JOIN graph_edges e2    ON e2.src_id   = n_cause.id
            JOIN graph_nodes n_fix ON n_fix.id    = e2.dst_id AND n_fix.type   = 'fix'
            WHERE n_sym.type = 'symptom' AND n_sym.name ILIKE %s
            LIMIT 10
            """,
            params,
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "content": f"Symptom: {r[0]} → Cause: {r[2]} → Fix: {r[4]}",
            "symptom": r[0],
            "cause": r[2],
            "fix": r[4],
        }
        for r in rows
    ]


def source_viewer(chunk_id: int) -> dict[str, Any]:
    """Return the source passage for a given chunk id (for inline citation display)."""
    from config import get_connection

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT content, source_refs, metadata FROM chunks WHERE id = %s",
            (chunk_id,),
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return {"error": f"chunk {chunk_id} not found", "content": ""}

    content, source_refs_raw, metadata_raw = row
    return {
        "chunk_id": chunk_id,
        "content": content,
        "source_refs": (
            json.loads(source_refs_raw) if isinstance(source_refs_raw, str) else source_refs_raw
        ),
        "metadata": (
            json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
        ),
    }


# Registry exposed to the LLM as callable tool schemas
TOOLS = {
    "hybrid_search": hybrid_search,
    "graph_lookup": graph_lookup,
    "spec_lookup": spec_lookup,
    "conflict_check": conflict_check,
    "unit_convert": unit_convert,
    "diagnostic_tree": diagnostic_tree,
    "source_viewer": source_viewer,
}
