"""Entity canonicalization and noise filtering for the knowledge graph."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from .kg_ontology import NODE_TYPES, SYSTEM_PARTS

_MONTHS = (
    "jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|"
    "aug|august|sep|sept|september|oct|october|nov|november|dec|december|"
    "januar|februar|maerz|märz|april|mai|juni|juli|august|september|"
    "oktober|november|dezember"
)
_DATE_RE = re.compile(
    rf"^\s*(?:({_MONTHS})\s+\d{{4}}|\d{{4}}|\d{{1,2}}[./-]\d{{1,2}}[./-]\d{{2,4}})\s*$",
    re.IGNORECASE,
)
_PAGE_RE = re.compile(r"^\s*(?:p\.?|page|seite)\s*\d+\s*$", re.IGNORECASE)
_SECTION_RE = re.compile(r"^\s*\d+(?:\.\d+){1,5}\s*$")
_STANDALONE_NUM_RE = re.compile(r"^\s*[+-]?\d+(?:[.,]\d+)?\s*$")
_UNIT_RE = re.compile(r"^\s*(?:°c|°f|c|f|k|%|rpm|nm|bar|psi|kw|hp|v|a|hz|s|min|mm|kg|l)\s*$", re.IGNORECASE)
_SPEC_RE = re.compile(
    r"^\s*(?:±\s*)?[+-]?\d+(?:[.,]\d+)?\s*(?:hours?|hrs?|h|°c|°f|%|rpm|nm|bar|psi|kw|hp|v|a|hz|s|min|mm|kg|l)\s*$",
    re.IGNORECASE,
)
_METRIC_THREAD_RE = re.compile(r"^\s*M\d+(?:[xX]\d+(?:[.,]\d+)?)?\s*$")
_ENGINE_RE = re.compile(r"\b(?:hirth\s*)?\d{3,4}[a-z]?\b", re.IGNORECASE)

_NOISE_PHRASES = (
    "copyright",
    "all rights reserved",
    "table of contents",
    "inhaltsverzeichnis",
    "contents",
    "confidential",
    "distribution",
    "revision",
)

_SYSTEM_ALIASES = {
    "ignition": "Ignition System",
    "ignition system": "Ignition System",
    "zündung": "Ignition System",
    "fuel": "Fuel System",
    "fuel system": "Fuel System",
    "kraftstoff": "Fuel System",
    "air intake": "Air Intake System",
    "intake": "Air Intake System",
    "ansaug": "Air Intake System",
    "exhaust": "Exhaust System",
    "auspuff": "Exhaust System",
    "crankcase": "Crankcase",
    "kurbelgehäuse": "Crankcase",
    "lubrication": "Lubrication System",
    "cooling": "Cooling System",
    "certification": "Certification / Testing",
    "testing": "Certification / Testing",
    "maintenance": "Maintenance",
}

_PART_ALIASES = {
    "spark plug": "Spark Plug",
    "plug": "Spark Plug",
    "sparking plug": "Spark Plug",
    "zündkerze": "Spark Plug",
    "carburetor": "Carburetor",
    "carburettor": "Carburetor",
    "carb": "Carburetor",
    "vergaser": "Carburetor",
    "fuel filter": "Fuel Filter",
    "filter fuel": "Fuel Filter",
    "kraftstofffilter": "Fuel Filter",
    "fuel pump": "Fuel Pump",
    "kraftstoffpumpe": "Fuel Pump",
    "fuel line": "Fuel Line",
    "reed valve": "Reed Valve",
    "air filter": "Air Filter",
    "luftfilter": "Air Filter",
    "muffler": "Muffler",
    "silencer": "Muffler",
    "exhaust muffler": "Muffler",
    "exhaust manifold": "Exhaust Manifold",
    "ignition coil": "Ignition Coil",
    "coil": "Ignition Coil",
    "zündspule": "Ignition Coil",
    "cdi": "CDI / Ignition Module",
    "ignition module": "CDI / Ignition Module",
    "piston": "Piston",
    "kolben": "Piston",
    "piston ring": "Piston Ring",
    "ring": "Piston Ring",
    "cylinder": "Cylinder Head",
    "zylinder": "Cylinder Head",
    "cylinder head": "Cylinder Head",
    "zylinderkopf": "Cylinder Head",
    "crankshaft": "Crankshaft",
    "crankcase seal": "Crankcase Seal",
    "oil injection pump": "Oil Injection Pump",
    "cooling fan": "Cooling Fan",
    "temperature sensor": "Temperature Sensor",
}

_SYMPTOM_ALIASES = {
    "won't start": "Won't Start",
    "will not start": "Won't Start",
    "hard starting": "Hard Starting",
    "misfire": "Misfire",
    "rough idle": "Rough Idle",
    "loss of power": "Loss of Power",
    "overheating": "Overheating",
    "excessive smoke": "Excessive Smoke",
    "detonation": "Detonation / Knocking",
    "knocking": "Detonation / Knocking",
    "high egt": "High EGT",
    "fuel starvation": "Fuel Starvation",
    "poor compression": "Poor Compression",
}

_CAUSE_ALIASES = {
    "spark plug fouling": "Spark Plug Fouling",
    "weak spark": "Weak Spark",
    "incorrect mixture": "Incorrect Mixture",
    "fuel restriction": "Fuel Restriction",
    "air leak": "Air Leak",
    "blocked jet": "Blocked Jet",
    "clogged jets": "Blocked Jet",
    "low compression": "Low Compression",
    "overheating": "Overheating",
    "incorrect timing": "Incorrect Timing",
    "carbon buildup": "Carbon Buildup",
}

_FIX_ALIASES = {
    "clean spark plug": "Clean Spark Plug",
    "replace spark plug": "Replace Spark Plug",
    "check ignition": "Check Ignition",
    "clean carburetor": "Clean Carburetor",
    "replace fuel filter": "Replace Fuel Filter",
    "inspect fuel lines": "Inspect Fuel Lines",
    "adjust mixture": "Adjust Mixture",
    "check compression": "Check Compression",
    "inspect cooling system": "Inspect Cooling System",
    "decarbonize exhaust": "Decarbonize Exhaust",
}

_CANONICAL_BY_TYPE = {
    "system": _SYSTEM_ALIASES,
    "part": _PART_ALIASES,
    "symptom": _SYMPTOM_ALIASES,
    "cause": _CAUSE_ALIASES,
    "fix": _FIX_ALIASES,
}

_ALIASES_BY_CANONICAL: dict[str, list[str]] = {}
for aliases in _CANONICAL_BY_TYPE.values():
    for alias, canonical in aliases.items():
        _ALIASES_BY_CANONICAL.setdefault(canonical, []).append(alias)

_SYSTEM_NAMES = {name.casefold(): name for name in SYSTEM_PARTS}
_PART_NAMES = {
    part.casefold(): part
    for parts in SYSTEM_PARTS.values()
    for part in parts
}


def normalize_entity(raw_text: str, proposed_type: str | None = None) -> dict[str, Any]:
    raw = _clean(raw_text)
    if not raw:
        return _reject(raw_text, proposed_type, "empty")

    if _is_noise(raw):
        return _reject(raw, proposed_type, "noise")

    proposed = _clean_type(proposed_type)
    if _is_spec(raw):
        return _result(raw, raw, "spec", [raw])

    lookup_key = _lookup_key(raw)
    for entity_type in _preferred_types(proposed):
        canonical = _CANONICAL_BY_TYPE.get(entity_type, {}).get(lookup_key)
        if canonical:
            return _result(canonical, canonical, entity_type, _ALIASES_BY_CANONICAL.get(canonical, [raw]))

    if lookup_key in _SYSTEM_NAMES:
        canonical = _SYSTEM_NAMES[lookup_key]
        return _result(canonical, canonical, "system", _ALIASES_BY_CANONICAL.get(canonical, [raw]))

    if lookup_key in _PART_NAMES:
        canonical = _PART_NAMES[lookup_key]
        return _result(canonical, canonical, "part", _ALIASES_BY_CANONICAL.get(canonical, [raw]))

    classified = classify_entity_type(raw)
    entity_type = proposed if proposed != "unknown" else classified
    if entity_type == "engine" and not _looks_like_engine(raw):
        return _reject(raw, proposed_type, "not_engine")

    display = _display_name(raw, entity_type)
    return _result(display, display, entity_type, [raw])


def classify_entity_type(text: str, context: str = "") -> str:
    raw = _clean(text)
    if not raw or _is_noise(raw):
        return "unknown"
    if _is_spec(raw):
        return "spec"

    key = _lookup_key(raw)
    for entity_type, aliases in _CANONICAL_BY_TYPE.items():
        if key in aliases:
            return entity_type
    if key in _SYSTEM_NAMES:
        return "system"
    if key in _PART_NAMES:
        return "part"
    if _looks_like_engine(raw):
        return "engine"

    text_context = f"{raw} {context}".casefold()
    if any(word in text_context for word in ("procedure", "inspect", "check", "replace", "clean", "adjust")):
        return "procedure"
    if any(word in text_context for word in ("warning", "caution", "danger", "achtung")):
        return "warning"
    return "unknown"


def is_noise_entity(raw_text: str, proposed_type: str | None = None) -> bool:
    return not normalize_entity(raw_text, proposed_type).get("is_valid", False)


def _preferred_types(proposed: str) -> list[str]:
    ordered = ["part", "system", "symptom", "cause", "fix"]
    if proposed in _CANONICAL_BY_TYPE:
        return [proposed] + [t for t in ordered if t != proposed]
    return ordered


def _clean_type(value: str | None) -> str:
    lowered = (value or "unknown").strip().lower()
    return lowered if lowered in NODE_TYPES else "unknown"


def _clean(value: str | None) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n:;,.()[]{}")


def _lookup_key(value: str) -> str:
    text = _clean(value).casefold()
    text = text.replace("behaviour", "behavior")
    text = re.sub(r"[\-_]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _is_noise(value: str) -> bool:
    text = _clean(value)
    folded = text.casefold()
    folded_ascii = _strip_accents(folded)
    if len(folded) < 2:
        return True
    if _DATE_RE.match(folded) or _DATE_RE.match(folded_ascii):
        return True
    if _PAGE_RE.match(folded) or _SECTION_RE.match(folded):
        return True
    if _UNIT_RE.match(folded) or _STANDALONE_NUM_RE.match(folded):
        return True
    if any(phrase in folded for phrase in _NOISE_PHRASES):
        return True
    return False


def _is_spec(value: str) -> bool:
    return bool(_SPEC_RE.match(value) or _METRIC_THREAD_RE.match(value))


def _looks_like_engine(value: str) -> bool:
    folded = value.casefold()
    if "two-stroke engine" in folded or "two stroke engine" in folded:
        return True
    if "hirth" in folded and _ENGINE_RE.search(folded):
        return True
    return False


def _display_name(value: str, entity_type: str) -> str:
    if entity_type == "engine":
        return re.sub(r"\bhirth\s*", "Hirth ", value, flags=re.IGNORECASE).strip()
    if entity_type == "spec":
        return value
    return value[:1].upper() + value[1:]


def _result(canonical: str, display: str, entity_type: str, aliases: list[str]) -> dict[str, Any]:
    clean_aliases = sorted({_clean(alias) for alias in aliases if _clean(alias)}, key=str.casefold)
    return {
        "canonical_name": canonical,
        "display_name": display,
        "type": _clean_type(entity_type),
        "aliases": clean_aliases,
        "is_valid": True,
        "reject_reason": "",
    }


def _reject(raw: str, proposed_type: str | None, reason: str) -> dict[str, Any]:
    return {
        "canonical_name": "",
        "display_name": _clean(raw),
        "type": _clean_type(proposed_type),
        "aliases": [],
        "is_valid": False,
        "reject_reason": reason,
    }
