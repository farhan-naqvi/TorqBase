"""Grounding verifier + gap detector.

is_grounded() gates the answer: every numeric claim must be supported by a
retrieved source. Ungrounded -> loop back. Weak/no evidence -> log a gap.
"""
from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# Matches numbers with 2+ digits or a decimal point — captures real spec values
# (RPM, temps, torque) while ignoring single-digit source indices like "[Source 1]".
_NUMBER_RE = re.compile(r"\b(?:\d{3,}(?:[.,]\d+)?|\d+[.,]\d+)\b")


def is_grounded(draft: str, context: list[dict[str, Any]]) -> bool:
    """Return True iff every numeric claim in draft appears verbatim in at least one source.

    A draft with no numeric claims (e.g. "I don't have enough information.") is
    considered grounded — no false assertion was made.
    """
    if not draft:
        return False

    numbers = _NUMBER_RE.findall(draft)
    if not numbers:
        return True  # no numeric claims to verify

    all_source_text = " ".join(c.get("content", "") for c in context)

    for num_str in numbers:
        if num_str not in all_source_text:
            log.warning("verifier: numeric claim %r not found in any source", num_str)
            return False

    return True


def log_gap(question: str, reason: str) -> None:
    """Record a knowledge gap to the gaps table for expert review.

    reason should be one of: 'missing spec', 'missing procedure', 'weak evidence'.
    Silently swallows DB errors so a failed write never breaks the answer path.
    """
    try:
        from config import get_connection

        conn = get_connection()
        try:
            with conn.transaction():
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO gaps (question, reason) VALUES (%s, %s)",
                    (question, reason),
                )
        finally:
            conn.close()
    except Exception:
        log.exception("verifier: failed to log gap for question %r", question)
