"""Extract tags + entities (de/en) per chunk via the LLM, with regex fallback.

When the LLM is unavailable (Ollama not running), a lightweight regex extractor
ensures entities and tags are always populated from the text itself.
"""
from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

_MIN_CONTENT_LEN = 80
_MAX_CONTENT_LEN = 600

# ---------------------------------------------------------------------------
# Regex-based fallback extractor (no LLM required)
# ---------------------------------------------------------------------------

_ENGINE_RE   = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z]?\d{3,4}[A-Za-z]*))\b")
_SPEC_RE     = re.compile(r"\b(\d+\.?\d*\s*(?:rpm|bar|°C|°F|kW|Nm|cc|mm|kg|l|hp|psi|V|A|Hz|s|min))\b", re.IGNORECASE)
_PART_KEYWORDS = [
    "ignition", "carburetor", "carburettor", "cylinder", "piston", "crankshaft",
    "reed valve", "exhaust", "intake", "throttle", "spark plug", "gearbox",
    "bearing", "gasket", "flywheel", "magneto", "clutch", "needle valve",
    "jet", "venturi", "port", "transfer port", "boost port", "main jet",
    "fuel pump", "oil pump", "cooling fin", "head gasket", "connecting rod",
]
_PART_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _PART_KEYWORDS) + r")\b",
    re.IGNORECASE,
)
_TAG_KEYWORDS = [
    "endurance", "test", "power", "torque", "speed", "altitude", "pressure",
    "temperature", "vibration", "lubrication", "scavenging", "supercharger",
    "turbocharger", "two-stroke", "two stroke", "four-stroke", "diesel", "otto",
    "FAR", "EASA", "CS-E", "certification", "maintenance", "overhaul",
]
_TAG_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _TAG_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def _regex_extract(content: str) -> dict[str, Any]:
    """Fallback: extract entities and tags from text using regex patterns."""
    entities: list[dict[str, str]] = []
    seen: set[str] = set()

    for m in _ENGINE_RE.finditer(content):
        name = m.group(1).strip()
        key = name.lower()
        if key not in seen and any(c.isdigit() for c in name):
            entities.append({"type": "engine", "name": name})
            seen.add(key)

    for m in _PART_RE.finditer(content):
        name = m.group(1).lower()
        if name not in seen:
            entities.append({"type": "part", "name": name})
            seen.add(name)

    for m in _SPEC_RE.finditer(content):
        name = m.group(1).strip()
        key = name.lower()
        if key not in seen:
            entities.append({"type": "spec", "name": name})
            seen.add(key)

    tag_set: set[str] = set()
    for m in _TAG_RE.finditer(content):
        tag_set.add(m.group(1).lower())
    tags = sorted(tag_set)[:8]

    return {"entities": entities[:20], "tags": tags}


# ---------------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------------

def enrich(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach 'entities' and 'tags' to each chunk's metadata.

    Tries the LLM first; falls back to regex extraction when the LLM is
    unavailable so entities are always populated.
    """
    enriched: list[dict[str, Any]] = []
    for chunk in chunks:
        content = chunk.get("content", "")
        if len(content) < _MIN_CONTENT_LEN:
            enriched.append(chunk)
            continue

        result: dict[str, Any] | None = None

        # LLM extraction (best effort)
        try:
            from llm import chat_json

            result = chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "Extract named entities and topic tags from this two-stroke engine text. "
                            "Return JSON only: "
                            "{\"entities\": [{\"type\": \"engine|part|symptom|spec|unit\", \"name\": str}], "
                            "\"tags\": [str]}. "
                            "Be concise; only extract what is explicitly stated."
                        ),
                    },
                    {"role": "user", "content": content[:_MAX_CONTENT_LEN]},
                ],
                max_tokens=200,
            )
            if not isinstance(result, dict):
                result = None
        except Exception:
            log.debug("domain_enricher: LLM unavailable; using regex fallback")
            result = None

        # Regex fallback when LLM failed or returned bad data
        if result is None:
            result = _regex_extract(content)

        chunk = dict(chunk)
        chunk["metadata"] = dict(chunk.get("metadata") or {})
        chunk["metadata"]["entities"] = result.get("entities", [])
        chunk["metadata"]["tags"]     = result.get("tags", [])
        enriched.append(chunk)

    return enriched
