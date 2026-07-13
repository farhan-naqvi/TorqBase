"""Build the knowledge graph from KG-relevant document chunks.

The graph always receives a compact manual seed ontology first, then document
facts are extracted chunk-by-chunk with provenance attached to edges/nodes.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from .kg_normalizer import classify_entity_type, normalize_entity
from .kg_ontology import normalize_node_type, normalize_relation, seed_graph
from .kg_rules import extract_rules
from .types import ParsedDoc

log = logging.getLogger(__name__)

_MAX_LLM_CHUNK_LEN = 1200
_DEFAULT_MAX_CHUNKS = 40
_DEFAULT_MAX_LLM_CHUNKS = 0
_ALLOWED_EXTRACTION_METHODS = {"manual_seed", "rule", "regex", "table", "llm", "unknown"}

_ENGINE_RE = re.compile(r"\b(?:Hirth\s*)?\d{3,4}[A-Za-z]?\b", re.IGNORECASE)
_PART_KEYWORDS = [
    "ignition", "carburetor", "carburettor", "cylinder", "piston", "crankshaft",
    "reed valve", "exhaust", "intake", "throttle", "spark plug", "gearbox",
    "bearing", "gasket", "flywheel", "magneto", "clutch", "connecting rod",
    "fuel pump", "oil pump", "fuel filter", "fuel line", "air filter",
    "transfer port", "main jet", "muffler", "silencer", "temperature sensor",
]
_PART_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _PART_KEYWORDS) + r")\b",
    re.IGNORECASE,
)
_SPEC_RE = re.compile(
    r"\b(?:±\s*)?[+-]?\d+(?:[.,]\d+)?\s*(?:hours?|hrs?|h|rpm|bar|°C|Â°C|°F|Â°F|kW|Nm|cc|mm|kg|l|hp|psi|V|A|Hz|s|min|%)\b",
    re.IGNORECASE,
)
_KG_KEYWORDS = {
    "symptom": 3.0, "cause": 3.0, "remedy": 3.5, "fault": 3.0, "trouble": 3.0,
    "troubleshooting": 4.0, "maintenance": 2.5, "warning": 2.0, "caution": 2.0,
    "check": 2.0, "replace": 2.0, "clean": 2.0, "inspect": 2.0, "adjust": 2.0,
    "pressure": 1.5, "temperature": 1.5, "rpm": 1.5, "torque": 1.5,
    "hours": 1.5, "spark": 2.0, "fuel": 2.0, "carburetor": 2.0,
    "carburettor": 2.0, "overheating": 2.5, "misfire": 2.5, "exhaust": 1.5,
    "compression": 2.0, "certification": 1.5, "endurance": 1.5, "test": 1.0,
}


def score_chunk_for_kg(chunk_text: str, metadata: dict[str, Any] | None = None) -> float:
    """Score a chunk for likely diagnostic/engineering graph value."""
    metadata = metadata or {}
    text = (chunk_text or "").lower()
    if not text.strip():
        return 0.0

    score = 0.0
    for keyword, weight in _KG_KEYWORDS.items():
        if keyword in text:
            score += weight
    if metadata.get("chunk_type") == "table" or metadata.get("type") == "table":
        score += 3.0
    if _PART_RE.search(chunk_text):
        score += 2.0
    if _SPEC_RE.search(chunk_text):
        score += 2.0
    if _ENGINE_RE.search(chunk_text):
        score += 1.0
    return score


def extract(doc: ParsedDoc, chunks: list[dict[str, Any]] | None = None) -> dict[str, int]:
    """Extract KG facts from selected chunks and write to graph tables.

    Existing callers can still pass only ParsedDoc; new ingestion paths should
    pass the post-dedup chunk list so page/chunk provenance is available.
    """
    stats = {
        "chunks_considered": 0,
        "chunks_selected": 0,
        "nodes_extracted": 0,
        "edges_extracted": 0,
        "nodes_rejected": 0,
        "edges_rejected": 0,
    }

    seed_stats = _seed()
    stats["nodes_extracted"] += seed_stats["nodes_inserted"]
    stats["edges_extracted"] += seed_stats["edges_inserted"]

    selected = _select_chunks(doc, chunks)
    stats["chunks_considered"] = len(chunks) if chunks is not None else 1
    stats["chunks_selected"] = len(selected)

    use_llm = os.getenv("KG_USE_LLM", "").strip().lower() in {"1", "true", "yes"}
    max_llm_chunks = _env_int("KG_MAX_LLM_CHUNKS", _DEFAULT_MAX_LLM_CHUNKS)

    for idx, chunk in enumerate(selected):
        try:
            result = _extract_from_chunk(chunk)
            if use_llm and idx < max_llm_chunks:
                result = _merge_results(result, _llm_extract_chunk(chunk))

            upsert_stats = _upsert(result.get("nodes", []), result.get("edges", []))
            stats["nodes_extracted"] += upsert_stats["nodes_inserted"]
            stats["edges_extracted"] += upsert_stats["edges_inserted"]
            stats["nodes_rejected"] += upsert_stats["nodes_rejected"]
            stats["edges_rejected"] += upsert_stats["edges_rejected"]
        except Exception:
            log.exception("graph_builder: chunk extraction failed; continuing")

    log.info("graph_builder: %s", stats)
    return stats


def _select_chunks(doc: ParsedDoc, chunks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if chunks is None:
        return [_chunk_with_provenance(doc.text, dict(doc.metadata), [doc.source_ref])]

    max_chunks = _env_int("KG_MAX_CHUNKS_PER_DOC", _DEFAULT_MAX_CHUNKS)
    scored = [
        (score_chunk_for_kg(str(c.get("content", "")), dict(c.get("metadata") or {})), c)
        for c in chunks
        if str(c.get("content", "")).strip()
    ]
    if len(scored) <= max_chunks:
        return [c for _, c in scored]

    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    selected = [c for score, c in ranked[:max_chunks] if score > 0]
    return selected or [c for _, c in ranked[:5]]


def _extract_from_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    text = str(chunk.get("content", ""))
    method = "table" if _is_table_chunk(chunk) else "rule"
    confidence = 0.9 if method == "table" else 0.78
    props = _chunk_props(chunk, method, confidence)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[tuple[str, str]] = set()

    def add_node(ntype: str, name: str) -> str | None:
        normalized = normalize_entity(name, ntype)
        if not normalized["is_valid"]:
            return None
        key = (normalized["type"], normalized["canonical_name"])
        if key not in seen_nodes:
            seen_nodes.add(key)
            nodes.append({
                "type": normalized["type"],
                "name": normalized["canonical_name"],
                "props": dict(props),
            })
        return normalized["canonical_name"]

    for match in _ENGINE_RE.finditer(text):
        raw = match.group(0)
        if "hirth" not in raw.lower() and re.fullmatch(r"\d{3,4}[A-Za-z]?", raw):
            raw = f"Hirth {raw}"
        add_node("engine", raw)

    for match in _PART_RE.finditer(text):
        add_node("part", match.group(1))

    for match in _SPEC_RE.finditer(text):
        add_node("spec", match.group(0))

    for entity in dict(chunk.get("metadata") or {}).get("entities", []):
        add_node(str(entity.get("type", "unknown")), str(entity.get("name", "")))

    engine_names = [n["name"] for n in nodes if n["type"] == "engine"]
    part_names = [n["name"] for n in nodes if n["type"] == "part"]
    spec_names = [n["name"] for n in nodes if n["type"] == "spec"]

    for engine in engine_names:
        for part in part_names:
            edges.append(_edge(engine, part, "HAS_PART", "engine", "part", props))
        for spec in spec_names:
            edges.append(_edge(engine, spec, "HAS_SPEC", "engine", "spec", props))

    for part in part_names:
        for spec in spec_names:
            edges.append(_edge(part, spec, "HAS_SPEC", "part", "spec", props))

    action = _action_type(text)
    if action:
        procedure = add_node("procedure", _procedure_name(action, part_names))
        if procedure:
            for part in part_names:
                edges.append(_edge(part, procedure, "REQUIRES_PROCEDURE", "part", "procedure", props))

    if classify_entity_type("warning", text) == "warning" and _has_warning(text):
        warning = add_node("warning", "Technical Warning")
        if warning:
            for part in part_names:
                edges.append(_edge(part, warning, "RELATED_TO", "part", "warning", props))

    rule_result = extract_rules(chunk, _chunk_props)
    return _merge_results(rule_result, {"nodes": nodes, "edges": edges})


def _llm_extract_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    text = str(chunk.get("content", ""))[:_MAX_LLM_CHUNK_LEN].strip()
    if not text:
        return {"nodes": [], "edges": []}
    try:
        from llm import chat_json

        result = chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "Extract explicit two-stroke engine graph facts from this chunk. "
                        "Return JSON only: "
                        "{\"nodes\": [{\"type\": \"engine|system|part|symptom|cause|fix|spec|procedure|test|warning\", \"name\": str}], "
                        "\"edges\": [{\"src\": str, \"dst\": str, \"relation\": str}]}. "
                        "Only include facts stated in the text."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=400,
        )
        if not isinstance(result, dict):
            return {"nodes": [], "edges": []}
        props = _chunk_props(chunk, "llm", 0.65)
        for node in result.get("nodes", []):
            node["props"] = {**props, **dict(node.get("props") or {})}
        for edge in result.get("edges", []):
            edge["props"] = {**props, **dict(edge.get("props") or {})}
        return result
    except Exception:
        log.debug("graph_builder: LLM chunk extraction unavailable")
        return {"nodes": [], "edges": []}


def _upsert(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, int]:
    """Upsert nodes then edges into graph_nodes / graph_edges."""
    from config import get_connection

    stats = {"nodes_inserted": 0, "edges_inserted": 0, "nodes_rejected": 0, "edges_rejected": 0}
    if not nodes:
        return stats

    normalized_nodes: list[dict[str, Any]] = []
    raw_to_keys: dict[str, list[tuple[str, str]]] = {}
    seen_node_keys: set[tuple[str, str]] = set()

    for node in nodes:
        raw_name = str(node.get("name", "")).strip()
        proposed_type = str(node.get("type", "")).strip()
        normalized = normalize_entity(raw_name, proposed_type)
        if not normalized["is_valid"]:
            stats["nodes_rejected"] += 1
            continue

        ntype = normalize_node_type(normalized["type"])
        nname = normalized["canonical_name"]
        key = (ntype, nname)
        raw_to_keys.setdefault(raw_name, []).append(key)
        if key in seen_node_keys:
            continue
        seen_node_keys.add(key)

        aliases = sorted(set(normalized["aliases"] + [raw_name]), key=str.casefold)
        props = dict(node.get("props") or {})
        props["aliases"] = aliases
        props["display_name"] = normalized["display_name"]
        normalized_nodes.append({"type": ntype, "name": nname, "props": props})

    if not normalized_nodes:
        return stats

    conn = get_connection()
    try:
        with conn.transaction():
            cur = conn.cursor()
            node_ids: dict[str, int] = {}

            for node in normalized_nodes:
                ntype = normalize_node_type(str(node.get("type", "")))
                nname = str(node.get("name", "")).strip()
                props = dict(node.get("props") or {})
                if not ntype or not nname:
                    continue
                cur.execute(
                    """
                    INSERT INTO graph_nodes (type, name, props)
                    VALUES (%s, %s, %s::jsonb)
                    ON CONFLICT (type, name) DO UPDATE
                    SET props = graph_nodes.props || EXCLUDED.props
                    RETURNING id
                    """,
                    (ntype, nname, json.dumps(props)),
                )
                row = cur.fetchone()
                if row:
                    node_ids[f"{ntype}::{nname}"] = int(row[0])
                    stats["nodes_inserted"] += 1

            for edge in edges:
                src_name = str(edge.get("src", "")).strip()
                dst_name = str(edge.get("dst", "")).strip()
                relation = normalize_relation(str(edge.get("relation", "")))
                props = _validate_edge_props(dict(edge.get("props") or {}))
                src_key = _edge_endpoint_key(edge, "src", src_name, raw_to_keys)
                dst_key = _edge_endpoint_key(edge, "dst", dst_name, raw_to_keys)
                if not src_key or not dst_key:
                    stats["edges_rejected"] += 1
                    continue
                src_id = node_ids.get(f"{src_key[0]}::{src_key[1]}")
                dst_id = node_ids.get(f"{dst_key[0]}::{dst_key[1]}")
                if not src_id or not dst_id or not relation:
                    stats["edges_rejected"] += 1
                    continue

                edge_key = _edge_key(src_key, relation, dst_key, props)
                props["edge_key"] = edge_key
                cur.execute(
                    """
                    INSERT INTO graph_edges (src_id, dst_id, relation, props)
                    SELECT %s, %s, %s, %s::jsonb
                    WHERE NOT EXISTS (
                        SELECT 1 FROM graph_edges
                        WHERE props->>'edge_key' = %s
                           OR (
                               src_id = %s AND dst_id = %s AND relation = %s
                               AND COALESCE(props->>'doc_id', '') = COALESCE(%s, '')
                               AND COALESCE(props->>'source_chunk_id', '') = COALESCE(%s, '')
                           )
                    )
                    """,
                    (
                        src_id, dst_id, relation, json.dumps(props), edge_key,
                        src_id, dst_id, relation,
                        str(props.get("doc_id") or ""),
                        str(props.get("source_chunk_id") or ""),
                    ),
                )
                stats["edges_inserted"] += cur.rowcount
    finally:
        conn.close()
    return stats


def _seed() -> dict[str, int]:
    graph = seed_graph()
    return _upsert(graph["nodes"], graph["edges"])


def _edge_endpoint_key(
    edge: dict[str, Any],
    side: str,
    name: str,
    raw_to_keys: dict[str, list[tuple[str, str]]],
) -> tuple[str, str] | None:
    hinted_type = str(edge.get(f"{side}_type", "")).strip()
    if hinted_type:
        normalized = normalize_entity(name, hinted_type)
        if normalized["is_valid"]:
            return (normalize_node_type(normalized["type"]), normalized["canonical_name"])

    keys = raw_to_keys.get(name) or []
    if keys:
        return keys[0]

    normalized = normalize_entity(name)
    if normalized["is_valid"]:
        return (normalize_node_type(normalized["type"]), normalized["canonical_name"])
    return None


def make_edge_props(
    doc_id: str | None,
    chunk_id: Any,
    page: Any,
    evidence: str,
    confidence: float,
    extraction_method: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create validated graph-edge provenance props."""
    method = extraction_method if extraction_method in _ALLOWED_EXTRACTION_METHODS else "unknown"
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    props: dict[str, Any] = {
        "doc_id": doc_id,
        "source_chunk_id": chunk_id,
        "chunk_id": chunk_id,
        "page": page,
        "evidence": _clean_evidence(evidence),
        "confidence": conf,
        "extraction_method": method,
    }
    if extra:
        props.update(extra)
    return _validate_edge_props(props)


def _chunk_props(chunk: dict[str, Any], method: str, confidence: float) -> dict[str, Any]:
    metadata = dict(chunk.get("metadata") or {})
    source_refs = chunk.get("source_refs") or []
    ref = source_refs[0] if source_refs else {}
    doc_id = metadata.get("doc_id") or metadata.get("filename") or ref.get("doc_id") or ref.get("filename") or "unknown"
    chunk_id = metadata.get("chunk_id", metadata.get("chunk_index"))
    page = metadata.get("page") or ref.get("page")
    source_title = metadata.get("title") or metadata.get("filename") or ref.get("filename") or ref.get("source")
    extra = {"source_title": source_title}
    for key in ("topic", "relative_path", "file_type", "source_title"):
        value = metadata.get(key) or ref.get(key)
        if value:
            extra[key] = value
    return make_edge_props(
        doc_id=doc_id,
        chunk_id=chunk_id,
        page=page,
        evidence=str(chunk.get("content", "")),
        confidence=confidence,
        extraction_method=method,
        extra=extra,
    )


def _chunk_with_provenance(
    text: str,
    metadata: dict[str, Any],
    source_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {"content": text, "metadata": {**metadata, "chunk_index": 0}, "source_refs": source_refs}


def _edge(
    src: str,
    dst: str,
    relation: str,
    src_type: str,
    dst_type: str,
    props: dict[str, Any],
) -> dict[str, Any]:
    return {
        "src": src,
        "dst": dst,
        "src_type": src_type,
        "dst_type": dst_type,
        "relation": relation,
        "props": dict(props),
    }


def _edge_key(
    src_key: tuple[str, str],
    relation: str,
    dst_key: tuple[str, str],
    props: dict[str, Any],
) -> str:
    return "|".join([
        src_key[0],
        src_key[1].casefold(),
        relation,
        dst_key[0],
        dst_key[1].casefold(),
        str(props.get("doc_id") or ""),
        str(props.get("source_chunk_id") or ""),
    ])


def _merge_results(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "nodes": list(left.get("nodes", [])) + list(right.get("nodes", [])),
        "edges": list(left.get("edges", [])) + list(right.get("edges", [])),
    }


def _action_type(text: str) -> str | None:
    lowered = text.lower()
    for action in ("replace", "clean", "inspect", "check", "adjust"):
        if action in lowered:
            return action
    return None


def _procedure_name(action: str, parts: list[str]) -> str:
    if parts:
        return f"{action.title()} {parts[0]}"
    return f"{action.title()} Procedure"


def _has_warning(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in ("warning", "caution", "danger", "achtung"))


def _is_table_chunk(chunk: dict[str, Any]) -> bool:
    metadata = dict(chunk.get("metadata") or {})
    return metadata.get("chunk_type") == "table" or metadata.get("type") == "table"


def _clean_evidence(evidence: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(evidence or "")).strip()
    return cleaned[:300]


def _validate_edge_props(props: dict[str, Any]) -> dict[str, Any]:
    method = props.get("extraction_method")
    if method not in _ALLOWED_EXTRACTION_METHODS:
        props["extraction_method"] = "unknown"
    try:
        confidence = float(props.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    props["confidence"] = max(0.0, min(1.0, confidence))
    props["evidence"] = _clean_evidence(str(props.get("evidence", "")))
    return props


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default
