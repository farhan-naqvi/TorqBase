"""Lightweight cross-document numeric conflict detection."""
from __future__ import annotations

import re
from typing import Any

UNIT_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"((?:°|\?)?C|(?:°|\?)?F|bar|psi|rpm|RPM|Nm|kW|hp|mm|cm|kg|l|ml|V|A|Hz|%|:1)?",
    re.IGNORECASE,
)

UNIT_VALID_RANGES = {
    "°C": (50, 1500),
    "°F": (120, 2700),
    "bar": (0.5, 250),
    "psi": (7, 3600),
    "rpm": (500, 30000),
    "RPM": (500, 30000),
    "Nm": (1, 10000),
    "kW": (1, 5000),
    "hp": (1, 7000),
    "mm": (0.1, 2000),
    "cm": (0.1, 200),
    "kg": (0.1, 5000),
    "l": (0.01, 10000),
    "ml": (0.1, 100000),
    "V": (1, 1000),
    "A": (0.1, 10000),
    "Hz": (1, 100000),
    "%": (0, 100),
    ":1": (1, 50),
}

SPEC_CONTEXT_WORDS = [
    "max", "min", "maximum", "minimum", "limit", "rated", "nominal",
    "operating", "temperature", "pressure", "speed", "torque", "power",
    "spec", "value", "target", "design", "allowable", "peak", "typical",
    "range", "threshold", "set point", "boundary", "grenzwert", "nennwert",
    "betrieb", "auslegung", "maximal", "minimal",
]


def group_by_doc(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group retrieval results by source document."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results or []:
        if not isinstance(r, dict):
            continue
        metadata = r.get("metadata") or {}
        refs = r.get("source_refs") or []
        ref = refs[0] if refs and isinstance(refs[0], dict) else {}
        doc_id = r.get("doc_id") or metadata.get("doc_id") or ref.get("doc_id") or "unknown"
        groups.setdefault(str(doc_id), []).append(r)
    return groups


def detect_conflicts(grouped: dict[str, list[dict[str, Any]]], question: str) -> list[dict[str, Any]]:
    """Find different numeric values with the same unit across multiple documents."""
    doc_values: dict[str, list[dict[str, Any]]] = {}
    for doc_id, chunks in (grouped or {}).items():
        doc_values[doc_id] = []
        for chunk in chunks:
            content = str(chunk.get("content") or chunk.get("value") or "")
            metadata = chunk.get("metadata") or {}
            refs = chunk.get("source_refs") or []
            ref = refs[0] if refs and isinstance(refs[0], dict) else {}
            filename = (
                chunk.get("filename")
                or metadata.get("source_title")
                or metadata.get("filename")
                or ref.get("filename")
                or doc_id
            )
            page = chunk.get("page") or metadata.get("page") or ref.get("page") or ""
            for match in UNIT_PATTERN.finditer(content):
                value = match.group(1).replace(",", ".")
                unit = _normalize_unit(match.group(2) or "")
                if not unit:
                    continue
                try:
                    fval = float(value)
                except ValueError:
                    continue
                start = max(0, match.start() - 80)
                end = min(len(content), match.end() + 80)
                snippet = content[start:end].replace("\n", " ").strip()
                valid_range = UNIT_VALID_RANGES.get(unit)
                if valid_range is None:
                    continue
                lo, hi = valid_range
                if not (lo <= fval <= hi):
                    continue
                if not any(word in snippet.lower() for word in SPEC_CONTEXT_WORDS):
                    continue
                doc_values[doc_id].append({
                    "value": value,
                    "unit": unit,
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": page,
                    "snippet": snippet,
                })

    doc_values = {doc_id: vals for doc_id, vals in doc_values.items() if vals}
    if len(doc_values) < 2:
        return []

    unit_to_doc_values: dict[str, dict[str, set[str]]] = {}
    unit_to_snippets: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for doc_id, vals in doc_values.items():
        for v in vals:
            unit = str(v["unit"]).lower()
            unit_to_doc_values.setdefault(unit, {}).setdefault(doc_id, set()).add(str(v["value"]))
            unit_to_snippets.setdefault(unit, {}).setdefault(doc_id, []).append(v)

    conflicts: list[dict[str, Any]] = []
    for unit, doc_map in unit_to_doc_values.items():
        if len(doc_map) < 2:
            continue
        flat_vals = {value for values in doc_map.values() for value in values}
        if len(flat_vals) <= 1:
            continue

        value_entries: list[dict[str, Any]] = []
        for doc_id in doc_map:
            value_entries.extend(unit_to_snippets[unit][doc_id][:2])

        numeric_vals = []
        for value in value_entries:
            try:
                numeric_vals.append(float(value["value"]))
            except ValueError:
                pass
        if not numeric_vals:
            continue
        spread = max(numeric_vals) - min(numeric_vals)
        pct_spread = spread / max(numeric_vals) if max(numeric_vals) else 0
        severity = "conflict" if spread >= 5 or pct_spread > 0.03 else "range"
        if severity != "conflict":
            continue
        conflicts.append({
            "unit": value_entries[0]["unit"],
            "values": value_entries,
            "severity": severity,
            "docs_involved": list(doc_map.keys()),
        })

    return conflicts[:3]


def _normalize_unit(unit: str) -> str:
    unit = (unit or "").strip()
    if unit.lower() in {"c", "?c", "°c"}:
        return "°C"
    if unit.lower() in {"f", "?f", "°f"}:
        return "°F"
    return unit
