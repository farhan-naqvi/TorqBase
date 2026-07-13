"""Deterministic KG relationship extraction for diagnostics and specs."""
from __future__ import annotations

import re
from typing import Any

from .kg_normalizer import normalize_entity

_MAX_ENTITY_LEN = 80
_VAGUE = {
    "the engine", "engine", "this manual", "manual", "system", "the system",
    "component", "components", "unit", "operation", "condition",
}
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_CAUSE_PATTERNS = [
    re.compile(r"(?P<cause>[^.;:\n]{3,80}?)\s+(?:may\s+)?causes?\s+(?P<symptom>[^.;:\n]{3,80})", re.I),
    re.compile(r"(?P<symptom>[^.;:\n]{3,80}?)\s+(?:may\s+be\s+|is\s+)?caused\s+by\s+(?P<cause>[^.;:\n]{3,80})", re.I),
    re.compile(r"(?:possible\s+cause|cause)\s*:\s*(?P<cause>[^.;\n]{3,80})", re.I),
]
_FIX_PATTERNS = [
    re.compile(r"\b(?P<action>check|inspect|clean|replace|adjust|tighten|remove|install)\s+(?P<target>[^.;,\n]{2,70})", re.I),
    re.compile(r"(?:remedy|corrective\s+action|correction|solution)\s*:\s*(?P<fix>[^.;\n]{3,90})", re.I),
]
_INTERVAL_RE = re.compile(
    r"\b(?P<interval>every\s+\d+\s+hours?|after\s+\d+\s+hours?|before\s+each\s+flight|daily\s+inspection|annual\s+inspection)\b",
    re.I,
)
_SPEC_RE = re.compile(
    r"\b(?:±\s*)?[+-]?\d+(?:[.,]\d+)?\s*(?:rpm|bar|psi|°C|Â°C|°F|Â°F|Nm|mm|hours?|hrs?|h|%|V|A|kg|g/h|l/h)\b",
    re.I,
)
_WARNING_RE = re.compile(r"\b(warning|caution|danger|do not|must not|achtung)\b", re.I)
_PART_RE = re.compile(
    r"\b(spark plug|plug|carburetor|carburettor|vergaser|fuel filter|fuel pump|fuel line|"
    r"air filter|reed valve|exhaust|muffler|silencer|piston ring|piston|cylinder head|"
    r"cylinder|ignition coil|coil|crankshaft|temperature sensor)\b",
    re.I,
)
_TROUBLE_HEADER_RE = re.compile(
    r"symptom|fault|problem|cause|possible cause|reason|remedy|correction|solution",
    re.I,
)


def extract_rules(chunk: dict[str, Any], make_props) -> dict[str, Any]:
    """Return graph_builder-compatible nodes/edges/debug from deterministic rules."""
    text = str(chunk.get("content", ""))
    base_props = make_props(chunk, "rule", 0.82)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    debug = {"rules_fired": 0, "nodes_rejected": 0, "edges_rejected": 0}
    seen_nodes: set[tuple[str, str]] = set()
    last_problem: str | None = None
    last_cause: str | None = None

    def add_node(ntype: str, raw: str, props: dict[str, Any] | None = None) -> str | None:
        cleaned = _clean_entity(raw)
        if not _is_useful_entity(cleaned):
            debug["nodes_rejected"] += 1
            return None
        normalized = normalize_entity(cleaned, ntype)
        if not normalized["is_valid"]:
            debug["nodes_rejected"] += 1
            return None
        key = (normalized["type"], normalized["canonical_name"])
        if key not in seen_nodes:
            seen_nodes.add(key)
            nodes.append({
                "type": normalized["type"],
                "name": normalized["canonical_name"],
                "props": dict(props or base_props),
            })
        return normalized["canonical_name"]

    def add_edge(src: str | None, dst: str | None, relation: str, src_type: str, dst_type: str, props: dict[str, Any] | None = None) -> None:
        if not src or not dst:
            debug["edges_rejected"] += 1
            return
        edges.append({
            "src": src,
            "dst": dst,
            "src_type": src_type,
            "dst_type": dst_type,
            "relation": relation,
            "props": dict(props or base_props),
        })
        debug["rules_fired"] += 1

    for sentence in _sentences(text):
        sentence_props = {**base_props, "evidence": _short(sentence)}
        cause_result = _extract_cause_sentence(sentence, add_node, add_edge, sentence_props)
        if cause_result:
            last_problem, last_cause = cause_result

        fix_names = _extract_fixes(sentence, add_node, add_edge, sentence_props, last_problem, last_cause)
        if fix_names and not last_cause and last_problem:
            last_cause = last_problem

        _extract_specs(sentence, add_node, add_edge, sentence_props)
        _extract_maintenance(sentence, add_node, add_edge, sentence_props)
        _extract_warning(sentence, add_node, add_edge, sentence_props)

    if _looks_like_troubleshooting_table(text):
        _extract_troubleshooting_table(text, add_node, add_edge, {**base_props, "extraction_method": "table", "confidence": 0.9})

    return {"nodes": nodes, "edges": edges, "debug": debug}


def _extract_cause_sentence(sentence, add_node, add_edge, props) -> tuple[str, str] | None:
    for pattern in _CAUSE_PATTERNS:
        match = pattern.search(sentence)
        if not match:
            continue
        cause_raw = match.groupdict().get("cause", "")
        symptom_raw = match.groupdict().get("symptom", "")
        symptom_raw = _infer_nearby_symptom(symptom_raw) or symptom_raw
        if not symptom_raw:
            symptom_raw = _infer_nearby_symptom(sentence) or "Diagnostic Symptom"
        symptom = add_node("symptom", symptom_raw, props)
        cause = add_node("cause", cause_raw, props)
        add_edge(symptom, cause, "CAUSED_BY", "symptom", "cause", props)
        return symptom, cause
    return None


def _extract_fixes(sentence, add_node, add_edge, props, last_problem, last_cause) -> list[str]:
    fixes: list[str] = []
    for pattern in _FIX_PATTERNS:
        for match in pattern.finditer(sentence):
            if "action" in match.groupdict() and match.groupdict().get("action"):
                actions = _actions_from_match(sentence, match.group("action"))
                target = _first_part(match.group("target")) or match.group("target")
                part_raw = _first_part(target)
                fix_candidates = [f"{action.title()} {target}" for action in actions]
            else:
                fix_candidates = [match.group("fix")]
                part_raw = _first_part(fix_candidates[0])
            for fix_raw in fix_candidates:
                fix = add_node("fix", fix_raw, props)
                fixes.append(fix) if fix else None
                if last_cause:
                    add_edge(last_cause, fix, "FIXED_BY", "cause", "fix", props)
                elif last_problem:
                    add_edge(last_problem, fix, "FIXED_BY", "symptom", "fix", props)
                if part_raw:
                    part = add_node("part", part_raw, props)
                    add_edge(fix, part, "RELATED_TO", "fix", "part", props)
    return fixes


def _extract_specs(sentence, add_node, add_edge, props) -> None:
    specs = [m.group(0) for m in _SPEC_RE.finditer(sentence)]
    if not specs:
        return
    parts = [_first_part(sentence)]
    parts = [p for p in parts if p]
    engines = re.findall(r"\bHirth\s*\d{3,4}[A-Za-z]?\b", sentence, flags=re.I)
    for spec_raw in specs:
        spec = add_node("spec", spec_raw, props)
        linked = False
        for part_raw in parts:
            part = add_node("part", part_raw, props)
            add_edge(part, spec, "HAS_SPEC", "part", "spec", props)
            linked = True
        for engine_raw in engines:
            engine = add_node("engine", engine_raw, props)
            add_edge(engine, spec, "HAS_SPEC", "engine", "spec", props)
            linked = True
        if not linked:
            source = add_node("source", "Source Chunk", props)
            add_edge(spec, source, "MENTIONED_IN", "spec", "source", props)


def _extract_maintenance(sentence, add_node, add_edge, props) -> None:
    match = _INTERVAL_RE.search(sentence)
    if not match:
        return
    interval = add_node("spec", match.group("interval"), props)
    part_raw = _first_part(sentence)
    owner = add_node("part", part_raw, props) if part_raw else add_node("system", "Maintenance", props)
    procedure = add_node("procedure", _procedure_from_sentence(sentence), props)
    add_edge(owner, procedure, "REQUIRES_PROCEDURE", "part" if part_raw else "system", "procedure", props)
    add_edge(procedure, interval, "HAS_SPEC", "procedure", "spec", props)


def _extract_warning(sentence, add_node, add_edge, props) -> None:
    if not _WARNING_RE.search(sentence):
        return
    warning = add_node("warning", "Technical Warning", props)
    part_raw = _first_part(sentence)
    if part_raw:
        part = add_node("part", part_raw, props)
        add_edge(part, warning, "RELATED_TO", "part", "warning", props)


def _extract_troubleshooting_table(text, add_node, add_edge, props) -> None:
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    header: list[str] | None = None
    for row in rows:
        cells = [c.strip() for c in re.split(r"\s*\|\s*|\t+", row) if c.strip()]
        if len(cells) < 3:
            continue
        if header is None and _TROUBLE_HEADER_RE.search(" ".join(cells)):
            header = [c.casefold() for c in cells]
            continue
        if header is None:
            continue
        symptom = add_node("symptom", cells[0], props)
        cause = add_node("cause", cells[1], props)
        fix = add_node("fix", cells[2], props)
        add_edge(symptom, cause, "CAUSED_BY", "symptom", "cause", props)
        add_edge(cause, fix, "FIXED_BY", "cause", "fix", props)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _clean_entity(text: str) -> str:
    cleaned = re.sub(r"^(?:possible\s+cause|cause|remedy|corrective\s+action|correction|solution)\s*:\s*", "", text, flags=re.I)
    inferred = _infer_nearby_symptom(cleaned)
    if inferred:
        return inferred
    cleaned = re.sub(r"\b(?:may|can|will|is|be|by|the|a|an|if|when|at|to|from)\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;")
    return cleaned[:_MAX_ENTITY_LEN]


def _is_useful_entity(text: str) -> bool:
    folded = text.casefold()
    return 2 <= len(text) <= _MAX_ENTITY_LEN and folded not in _VAGUE


def _first_part(text: str) -> str | None:
    match = _PART_RE.search(text)
    return match.group(1) if match else None


def _infer_nearby_symptom(sentence: str) -> str | None:
    for word in ("misfire", "overheating", "rough idle", "loss of power", "hard starting", "fuel starvation"):
        if word in sentence.lower():
            return word
    return None


def _procedure_from_sentence(sentence: str) -> str:
    action = "Inspect"
    for candidate in ("check", "inspect", "clean", "replace", "adjust", "tighten", "remove", "install"):
        if re.search(rf"\b{candidate}\b", sentence, re.I):
            action = candidate.title()
            break
    part = _first_part(sentence)
    return f"{action} {part}" if part else f"{action} Procedure"


def _actions_from_match(sentence: str, first_action: str) -> list[str]:
    actions = [first_action.lower()]
    for action in ("check", "inspect", "clean", "replace", "adjust", "tighten", "remove", "install"):
        if action not in actions and re.search(rf"\bor\s+{action}\b|\band\s+{action}\b", sentence, re.I):
            actions.append(action)
    return actions


def _looks_like_troubleshooting_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return len(lines) >= 2 and any("|" in line or "\t" in line for line in lines) and _TROUBLE_HEADER_RE.search(text)


def _short(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:300]
