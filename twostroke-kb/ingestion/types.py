"""Shared shapes for ingestion. Every parser returns a ParsedDoc so downstream
steps (corpus_builder, chunker) are parser-agnostic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Table:
    name: str | None
    rows: list[list[str]]
    units: dict[str, str] = field(default_factory=dict)  # column -> unit


@dataclass
class ParsedDoc:
    text: str                                   # extracted prose
    tables: list[Table] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)  # filename, page, type, lang...
    source_ref: dict[str, Any] = field(default_factory=dict)
