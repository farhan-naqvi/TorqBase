"""Windows .url shortcut -> store the target URL as a reference ONLY.

Scope rule: we do NOT fetch external content. Just record the link in metadata
so it can be surfaced ("see HYDAC International") and optionally ingested in phase 2.
"""
from __future__ import annotations

import configparser
from pathlib import Path

from ..types import ParsedDoc


def parse(path: str | Path) -> ParsedDoc:
    """Read the URL= line from the .url file and store it as a reference."""
    path = Path(path)
    url = ""
    try:
        cp = configparser.ConfigParser(interpolation=None)
        cp.read(path)
        url = cp.get("InternetShortcut", "URL", fallback="")
    except Exception:
        for line in path.read_text(errors="ignore").splitlines():
            if line.lower().startswith("url="):
                url = line.split("=", 1)[1].strip()
                break
    return ParsedDoc(
        text=f"Reference link: {url}",
        metadata={"type": "link", "url": url, "fetched": False, "filename": path.name},
        source_ref={"filename": path.name, "url": url},
    )
