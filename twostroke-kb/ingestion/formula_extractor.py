"""Lightweight formula detection from already-extracted text chunks."""
from __future__ import annotations

import re
from typing import Any

from ingestion.parsers.math_normalizer import normalize_math_unicode

_LATEX_RE = re.compile(
    r"(\\(?:frac|sum|int|sqrt|eta|Delta|theta|pi|alpha|beta|gamma|mu|lambda)\b[^\n.;]*)"
)
_INLINE_RE = re.compile(
    r"(?P<formula>\b[A-Z][A-Za-z0-9_]*|[A-Za-z]_[A-Za-z0-9]+|eta|delta|theta|lambda|mu|pi)\s*=\s*"
    r"(?P<expr>[A-Za-z0-9_.*\\/+\-^()\s]{3,140})"
)
_MARKER_RE = re.compile(r"\b(Equation|Eq\.|Gl\.|Formel|Formula)\s*[:.]?\s*\(?\d+[\w.-]*\)?", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def extract_formulas(text: str) -> list[dict[str, Any]]:
    """Return formula candidates with nearby explanatory context."""
    formulas: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not text:
        return formulas
    text = normalize_math_unicode(text)

    for match in _LATEX_RE.finditer(text):
        _add_formula(formulas, seen, text, match.start(), match.group(1).strip(), latex=True)

    for line_match in re.finditer(r"[^\n]+", text):
        line = line_match.group(0).strip()
        if not line:
            continue

        for match in _INLINE_RE.finditer(line):
            formula = f"{match.group('formula')} = {match.group('expr')}".strip(" ,.;")
            if _looks_like_formula(formula):
                _add_formula(formulas, seen, text, line_match.start() + match.start(), formula, latex=False)

        if "=" in line and len(line) <= 240 and _looks_like_formula(line):
            _add_formula(formulas, seen, text, line_match.start(), line.strip(" ,.;"), latex=False)

        if _MARKER_RE.search(line):
            formula = line[:240].strip()
            _add_formula(formulas, seen, text, line_match.start(), formula, latex=False)

    return formulas


def _looks_like_formula(value: str) -> bool:
    rhs = value.split("=", 1)[-1]
    return any(op in rhs for op in ("*", "/", "+", "-", "(", ")")) or bool(re.search(r"\d", rhs))


def _add_formula(
    formulas: list[dict[str, Any]],
    seen: set[str],
    text: str,
    start: int,
    formula: str,
    latex: bool,
) -> None:
    formula = re.sub(r"\s+", " ", normalize_math_unicode(formula)).strip()
    if len(formula) < 5 or formula in seen:
        return
    seen.add(formula)
    before, after = _context_around(text, start)
    formulas.append({
        "formula_text": formula,
        "formula_latex": formula if latex else None,
        "context_before": before,
        "context_after": after,
    })


def _context_around(text: str, start: int) -> tuple[str, str]:
    before_text = text[:start]
    after_text = text[start:]
    before_sentences = [s.strip() for s in _SENTENCE_RE.split(before_text) if s.strip()]
    after_sentences = [s.strip() for s in _SENTENCE_RE.split(after_text) if s.strip()]
    before = " ".join(before_sentences[-2:])[:600]
    after = " ".join(after_sentences[1:3] if len(after_sentences) > 1 else after_sentences[:1])[:600]
    return before, after
