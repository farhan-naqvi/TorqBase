"""Chunk prose into overlapping passages; keep tables intact as single chunks.

Page numbers are tracked via [PAGE_BREAK:N] sentinels inserted by pdf_parser.
Each prose chunk records source_refs[0]["page"] = the page it starts on.
Chunks extend to the nearest sentence boundary to avoid mid-sentence cuts.
"""
from __future__ import annotations

import re
from typing import Any

from .types import ParsedDoc, Table
from .parsers.math_normalizer import normalize_math_unicode

# Sentinel pattern inserted by pdf_parser between pages
_PAGE_BREAK_RE = re.compile(r"\[PAGE_BREAK:(\d+)\]")
_SLIDE_RE = re.compile(r"---\s*Slide\s+(\d+)\s*---", re.IGNORECASE)

# Sentence boundary: period/!/?  followed by whitespace or end-of-string
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)|\n\n")


def _table_to_text(table: Table) -> str:
    """Render a Table as a compact text block for embedding."""
    lines: list[str] = []
    if table.name:
        lines.append(f"[Table: {table.name}]")
    for row in table.rows:
        lines.append(" | ".join(cell.strip() for cell in row))
    if table.units:
        lines.append("Units: " + ", ".join(f"{k}={v}" for k, v in table.units.items()))
    return "\n".join(lines)


def _build_page_map(text: str) -> tuple[str, dict[int, int]]:
    """Strip [PAGE_BREAK:N] sentinels and return (clean_text, {char_offset: page_num}).

    char_offset is the position in the *clean* text where each new page begins.
    """
    page_map: dict[int, int] = {0: 1}  # text starts on page 1
    result: list[str] = []
    cursor = 0  # position in clean text

    for m in _PAGE_BREAK_RE.finditer(text):
        page_num = int(m.group(1))
        # Append text before this sentinel
        segment = text[cursor:m.start()]
        result.append(segment)
        cursor = m.end()
        # Record where the next page starts in clean text
        page_map[len("".join(result))] = page_num

    result.append(text[cursor:])
    return "".join(result), page_map


def _page_at(offset: int, page_map: dict[int, int]) -> int:
    """Return the page number active at the given char offset in clean text."""
    page = 1
    for start, num in sorted(page_map.items()):
        if start <= offset:
            page = num
        else:
            break
    return page


def _build_slide_map(text: str) -> dict[int, int]:
    """Return {char_offset: slide_num} for PPTX slide headers, if present."""
    return {m.start(): int(m.group(1)) for m in _SLIDE_RE.finditer(text)}


def _slide_at(offset: int, slide_map: dict[int, int]) -> int | None:
    slide = None
    for start, num in sorted(slide_map.items()):
        if start <= offset:
            slide = num
        else:
            break
    return slide


def _extend_to_sentence(text: str, end: int, max_extra: int = 150) -> int:
    """Extend end position forward to the nearest sentence boundary, up to max_extra chars."""
    limit = min(end + max_extra, len(text))
    search = text[end:limit]
    m = _SENTENCE_END_RE.search(search)
    if m:
        return end + m.end()
    return end


def chunk(doc: ParsedDoc, size: int = 700, overlap: int = 100) -> list[dict[str, Any]]:
    """Return a list of chunk dicts: {content, metadata, source_refs}.

    Prose is split by sliding window (size chars, overlap chars).
    Windows are extended to the nearest sentence boundary (up to 150 extra chars).
    Each Table becomes one chunk — never split.
    Page numbers from [PAGE_BREAK:N] sentinels are recorded in source_refs[0]["page"].
    """
    chunks: list[dict[str, Any]] = []
    idx = 0

    # Strip page sentinels and build offset→page map
    text, page_map = _build_page_map(normalize_math_unicode(doc.text))
    slide_map = _build_slide_map(text)
    base_ref = dict(doc.source_ref)  # shallow copy so we can add "page"

    step = size - overlap
    pos = 0
    while pos < len(text):
        raw_end = pos + size
        end = _extend_to_sentence(text, raw_end)  # snap to sentence boundary
        content = text[pos:end].strip()
        if content:
            page = _page_at(pos, page_map)
            slide = _slide_at(pos, slide_map)
            ref = {**base_ref, "page": page}
            if slide is not None:
                ref["slide"] = slide
            metadata = {**doc.metadata, "chunk_index": idx, "chunk_type": "prose", "page": page}
            if slide is not None:
                metadata["slide"] = slide
            chunks.append({
                "content": content,
                "metadata": metadata,
                "source_refs": [ref],
            })
            idx += 1
        pos += step
        if pos >= len(text):
            break

    # Table chunks — one per table, never split
    for table in doc.tables:
        content = normalize_math_unicode(_table_to_text(table))
        if content.strip():
            chunks.append({
                "content": content,
                "metadata": {**doc.metadata, "chunk_index": idx, "chunk_type": "table",
                              "table_name": table.name, "type": "table"},
                "source_refs": [doc.source_ref],
            })
            idx += 1

    return chunks
