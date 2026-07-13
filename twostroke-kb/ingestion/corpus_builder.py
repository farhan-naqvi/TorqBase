"""Normalize a ParsedDoc into clean text + table metadata ready for chunking."""
from __future__ import annotations

import re

from .types import ParsedDoc

# Lines that are purely decorative (dashes, underscores, equals, dots)
_BOILERPLATE_LINE = re.compile(r"^[\s\-_=.]{4,}$")


def normalize(doc: ParsedDoc) -> ParsedDoc:
    """Strip boilerplate lines, collapse whitespace, fill missing metadata keys.

    Returns a new ParsedDoc with cleaned text. Tables are passed through unchanged.
    """
    lines = doc.text.splitlines()
    cleaned_lines: list[str] = []
    for line in lines:
        if _BOILERPLATE_LINE.match(line):
            continue
        # collapse runs of spaces/tabs within a line
        cleaned_lines.append(re.sub(r"[ \t]{2,}", " ", line))

    # collapse 3+ consecutive blank lines down to 2
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines))
    # strip zero-width characters
    text = re.sub(r"[​‌‍﻿]", "", text)

    metadata = {
        "filename": doc.metadata.get("filename", "unknown"),
        "lang": doc.metadata.get("lang", "unknown"),
        "type": doc.metadata.get("type", "unknown"),
        **doc.metadata,  # keep any extras (pages, etc.)
    }

    return ParsedDoc(
        text=text,
        tables=doc.tables,
        images=doc.images,
        metadata=metadata,
        source_ref=doc.source_ref,
    )
