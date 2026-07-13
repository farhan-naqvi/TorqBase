"""Normalize PDF/PPTX math Unicode into stable plain-text formulas."""
from __future__ import annotations

import re
import unicodedata


def _chars(*codes: int) -> list[str]:
    return [chr(code) for code in codes]


_GREEK_MAP = {
    # Mathematical italic/bold italic Greek variants commonly emitted by PyMuPDF.
    chr(0x1D6FC): "alpha",
    chr(0x1D6FD): "beta",
    chr(0x1D6FE): "gamma",
    chr(0x1D6FF): "delta",
    chr(0x1D700): "epsilon",
    chr(0x1D701): "zeta",
    chr(0x1D702): "eta",
    chr(0x1D703): "theta",
    chr(0x1D705): "kappa",
    chr(0x1D706): "lambda",
    chr(0x1D707): "mu",
    chr(0x1D70B): "pi",
    chr(0x1D70C): "rho",
    chr(0x1D70E): "sigma",
    chr(0x1D714): "omega",
    chr(0x1D71F): "Delta",
    chr(0x1D72C): "Phi",
    chr(0x1D72E): "Sigma",
    chr(0x1D733): "Psi",
    chr(0x1D736): "alpha",
    chr(0x1D737): "beta",
    chr(0x1D738): "gamma",
    chr(0x1D739): "delta",
    chr(0x1D73A): "epsilon",
    chr(0x1D73B): "zeta",
    chr(0x1D73D): "theta",
    chr(0x1D73F): "kappa",
    chr(0x1D740): "lambda",
    chr(0x1D741): "mu",
    chr(0x1D745): "pi",
    chr(0x1D746): "rho",
    chr(0x1D748): "sigma",
    chr(0x1D74E): "omega",
    # Plain Greek variants, normalized to readable ASCII names for KaTeX/plain text.
    "η": "eta",
    "θ": "theta",
    "λ": "lambda",
    "μ": "mu",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "ω": "omega",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "κ": "kappa",
}

_SUBSCRIPT_MAP = {chr(0x2080 + i): f"_{i}" for i in range(10)}
_SUPERSCRIPT_MAP = {
    "\u00b2": "^2",
    "\u00b3": "^3",
    "\u00b9": "^1",
    **{chr(0x2070 + i): f"^{i}" for i in range(10)},
}
_OPERATOR_MAP = {
    "\u2219": "*",
    "\u00b7": "*",
    "\u22c5": "*",
    "\u00d7": "*",
    "\u2212": "-",
    "\u2215": "/",
}


def _math_alpha_to_ascii(ch: str) -> str | None:
    code = ord(ch)
    ranges = (
        (0x1D400, "A", 26), (0x1D41A, "a", 26),  # bold
        (0x1D434, "A", 26), (0x1D44E, "a", 26),  # italic
        (0x1D468, "A", 26), (0x1D482, "a", 26),  # bold italic
        (0x1D49C, "A", 26), (0x1D4B6, "a", 26),  # script
    )
    for start, base, count in ranges:
        if start <= code < start + count:
            return chr(ord(base) + code - start)
    return None


def normalize_math_unicode(text: str) -> str:
    """Return text with mathematical Unicode folded to readable formula text."""
    if not text:
        return ""

    mapped = "".join(
        _math_alpha_to_ascii(ch)
        or _GREEK_MAP.get(ch)
        or _SUBSCRIPT_MAP.get(ch)
        or _SUPERSCRIPT_MAP.get(ch)
        or _OPERATOR_MAP.get(ch)
        or ch
        for ch in text
    )
    normalized = unicodedata.normalize("NFKD", mapped)
    normalized = _formula_cleanup(normalized)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.splitlines()]
    return "\n".join(lines).strip()


def _formula_cleanup(text: str) -> str:
    """Small engineering-formula cleanups for common PDF extraction spacing."""
    text = re.sub(r"\b([A-Z])\s+([a-z])\b", r"\1_\2", text)
    text = re.sub(r"\b([A-Z])([a-z])\b", r"\1_\2", text)
    text = re.sub(r"\beta\s*e\b", "eta_e", text)
    text = re.sub(r"η\s*e\b", "eta_e", text)
    text = re.sub(r"\betae\b", "eta_e", text)
    text = re.sub(r"\bm\s+F\b", "m_dot_F", text)
    text = re.sub(r"\bm_F\b", "m_dot_F", text)
    text = re.sub(r"\bm_dot_F\s*\*\s*LHV\s+P_b\b", "m_dot_F * LHV / P_b", text)
    text = re.sub(r"\bBTE\s*\(\s*eta_e\s*\)\s*=\s*m_dot_F\s*\*\s*LHV\s+P_b\b", "BTE (eta_e) = m_dot_F * LHV / P_b", text)
    text = re.sub(r"\b([A-Za-z0-9_^)])\s*\*\s*([A-Za-z0-9_(])", r"\1 * \2", text)
    text = re.sub(r"\s*/\s*", " / ", text)
    return text
