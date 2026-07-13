"""KG-aware retrieval for diagnostic and engineering questions."""
from __future__ import annotations

import json
import re
from typing import Any

from ingestion.kg_normalizer import normalize_entity

_INTENT_TRIGGERS = {
    "diagnostic_cause": [
        "why", "cause", "caused by", "because", "overheating", "misfire",
        "rough idle", "loss of power", "won't start", "wont start", "hard start",
        "hard starting", "fuel starvation",
    ],
    "diagnostic_fix": [
        "fix", "troubleshoot", "repair", "remedy", "what should i check",
        "check", "replace", "clean", "inspect", "adjust",
    ],
    "part_lookup": [
        "related to", "part", "component", "spark plug", "carburetor",
        "carburettor", "fuel pump", "fuel filter", "piston", "exhaust",
    ],
    "spec_lookup": [
        "rpm", "temperature", "hours", "torque", "pressure", "bar", "psi",
        "°c", "nm", "spec", "specification",
    ],
    "procedure_lookup": [
        "procedure", "maintenance", "interval", "before each flight", "daily",
        "annual", "install", "remove",
    ],
}

_TERM_HINTS = [
    "misfire", "overheating", "rough idle", "loss of power", "hard starting",
    "won't start", "fuel starvation", "spark plug", "carburetor", "carburettor",
    "fuel pump", "fuel filter", "air filter", "piston", "piston ring",
    "cylinder head", "crankshaft", "exhaust", "muffler", "compression",
    "temperature", "rpm", "torque", "pressure",
]
_TERM_STOPWORDS = {
    "what", "should", "check", "inspect", "replace", "clean", "adjust",
    "engine", "does", "about", "related", "cause", "causes", "fix",
    "troubleshoot", "why", "hard", "start", "starting",
}


def classify_question_intent(question: str) -> dict[str, Any]:
    """Classify a user question into a lightweight KG retrieval intent."""
    q = question.casefold()
    scores: dict[str, int] = {}
    triggers: dict[str, list[str]] = {}
    for intent, terms in _INTENT_TRIGGERS.items():
        hits = [term for term in terms if term in q]
        if hits:
            scores[intent] = len(hits)
            triggers[intent] = hits
    if not scores:
        return {"intent": "general_question", "is_kg_relevant": False, "triggers": []}
    intent = max(scores, key=scores.get)
    return {"intent": intent, "is_kg_relevant": True, "triggers": triggers[intent]}


def retrieve_kg_context(question: str, limit: int = 5) -> dict[str, Any]:
    """Retrieve compact KG paths/neighborhoods relevant to the question."""
    intent = classify_question_intent(question)
    if not intent["is_kg_relevant"]:
        return {"intent": intent, "paths": [], "graph_evidence": [], "context": ""}

    terms = _query_terms(question)
    if not terms:
        return {"intent": intent, "paths": [], "graph_evidence": [], "context": ""}

    paths: list[dict[str, Any]] = []
    try:
        paths.extend(_diagnostic_paths(terms, limit=limit * 5))
        paths.extend(_neighborhood_paths(terms, limit=limit * 3))
    except Exception:
        paths = []

    paths = _dedupe_paths(paths, terms)[:limit]
    context = format_kg_context(paths)
    return {
        "intent": intent,
        "paths": paths,
        "graph_evidence": paths,
        "context": context,
    }


def format_kg_context(paths: list[dict[str, Any]]) -> str:
    if not paths:
        return ""
    lines = ["Knowledge Graph evidence:"]
    for idx, path in enumerate(paths, 1):
        lines.append(f"Path {idx}:")
        lines.append(path["path"])
        evidence = path.get("evidence") or ""
        if evidence:
            lines.append(f'Evidence: "{evidence}"')
        source = _source_label(path)
        if source:
            lines.append(f"Source: {source}")
        if path.get("confidence") is not None:
            lines.append(f"Confidence: {path['confidence']:.2f}")
    return "\n".join(lines)


def _query_terms(question: str) -> list[str]:
    q = question.casefold()
    terms = [hint for hint in _TERM_HINTS if hint in q]
    if "misfiring" in q:
        terms.append("misfire")
    if "won't start" in q or "wont start" in q:
        terms.append("Won't Start")
    words = [
        w for w in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{3,}", question.casefold())
        if w not in _TERM_STOPWORDS
    ]
    terms.extend(words[:8])

    canonical_terms: list[str] = []
    for term in terms:
        for proposed in ("symptom", "part", "cause", "fix", "spec", "procedure"):
            normalized = normalize_entity(term, proposed)
            if normalized["is_valid"]:
                canonical_terms.append(normalized["canonical_name"])
                break
    all_terms = terms + canonical_terms
    deduped: list[str] = []
    seen = set()
    for term in all_terms:
        key = term.casefold()
        if key not in seen and len(term.strip()) >= 3:
            seen.add(key)
            deduped.append(term.strip())
    return deduped[:10]


def _diagnostic_paths(terms: list[str], limit: int) -> list[dict[str, Any]]:
    from config import get_connection

    where, params = _match_clause("n_sym", terms, "n_cause", "n_fix")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                n_sym.name, e1.relation, n_cause.name, e1.props,
                e2.relation, n_fix.name, e2.props
            FROM graph_nodes n_sym
            JOIN graph_edges e1 ON e1.src_id = n_sym.id AND e1.relation = 'CAUSED_BY'
            JOIN graph_nodes n_cause ON n_cause.id = e1.dst_id
            LEFT JOIN graph_edges e2 ON e2.src_id = n_cause.id AND e2.relation = 'FIXED_BY'
            LEFT JOIN graph_nodes n_fix ON n_fix.id = e2.dst_id
            WHERE {where}
            LIMIT %s
            """,
            (*params, limit),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    result = []
    for sym, rel1, cause, props1, rel2, fix, props2 in rows:
        props = _props(props2) if fix else _props(props1)
        path = f"{sym} --{rel1}--> {cause}"
        if fix:
            path += f" --{rel2}--> {fix}"
        result.append(_path_dict(path, props, "diagnostic_path"))
    return result


def _neighborhood_paths(terms: list[str], limit: int) -> list[dict[str, Any]]:
    from config import get_connection

    where, params = _match_clause("n1", terms, "n2")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT n1.name, n1.type, e.relation, n2.name, n2.type, e.props
            FROM graph_nodes n1
            JOIN graph_edges e ON e.src_id = n1.id
            JOIN graph_nodes n2 ON n2.id = e.dst_id
            WHERE {where}
            LIMIT %s
            """,
            (*params, limit * 2),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return [
        _path_dict(f"{src} ({src_type}) --{rel}--> {dst} ({dst_type})", _props(props), "neighborhood")
        for src, src_type, rel, dst, dst_type, props in rows
    ]


def _match_clause(first_alias: str, terms: list[str], *extra_aliases: str) -> tuple[str, list[str]]:
    aliases = (first_alias,) + extra_aliases
    pieces = []
    params: list[str] = []
    for alias in aliases:
        for term in terms:
            pattern = f"%{term}%"
            pieces.append(f"({alias}.name ILIKE %s OR {alias}.props::text ILIKE %s)")
            params.extend([pattern, pattern])
    return " OR ".join(pieces) or "false", params


def _path_dict(path: str, props: dict[str, Any], kind: str) -> dict[str, Any]:
    confidence = props.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = None
    return {
        "kind": kind,
        "path": path,
        "evidence": str(props.get("evidence") or "")[:300],
        "confidence": confidence,
        "doc_id": props.get("doc_id"),
        "source_chunk_id": props.get("source_chunk_id", props.get("chunk_id")),
        "page": props.get("page"),
        "extraction_method": props.get("extraction_method", "unknown"),
        "source_title": props.get("source_title"),
    }


def _props(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {}
    return {}


def _dedupe_paths(paths: list[dict[str, Any]], terms: list[str]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for path in paths:
        key = path["path"]
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return sorted(
        result,
        key=lambda p: (_term_score(p["path"], terms), p.get("confidence") or 0.0),
        reverse=True,
    )


def _term_score(path: str, terms: list[str]) -> int:
    folded = path.casefold()
    return sum(1 for term in terms if term.casefold() in folded)


def _source_label(path: dict[str, Any]) -> str:
    title = path.get("source_title") or path.get("doc_id")
    page = path.get("page")
    chunk = path.get("source_chunk_id")
    parts = []
    if title:
        parts.append(str(title))
    if page:
        parts.append(f"page {page}")
    if chunk is not None:
        parts.append(f"chunk {chunk}")
    return ", ".join(parts)
