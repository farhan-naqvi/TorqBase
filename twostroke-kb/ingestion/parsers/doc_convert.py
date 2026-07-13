"""Legacy Word (.doc) -> convert to .docx/text first, then parse.

python-docx CANNOT read old binary .doc. Convert via LibreOffice (headless) or antiword.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from ..types import ParsedDoc
from . import docx_parser


def parse(path: str | Path) -> ParsedDoc:
    """Convert a legacy .doc to parseable form and return a ParsedDoc.

    Tries LibreOffice first (preserves tables/structure), falls back to antiword
    (text only). Raises RuntimeError if neither system binary is on PATH.
    """
    path = Path(path)

    if shutil.which("libreoffice"):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "docx",
                 "--outdir", tmp, str(path)],
                check=True,
                capture_output=True,
            )
            converted = Path(tmp) / (path.stem + ".docx")
            if converted.exists():
                doc = docx_parser.parse(converted)
                doc.metadata["filename"] = path.name
                doc.metadata["type"] = "doc"
                doc.source_ref["filename"] = path.name
                return doc

    if shutil.which("antiword"):
        text = subprocess.check_output(["antiword", str(path)], text=True)
        return ParsedDoc(
            text=text,
            metadata={"filename": path.name, "type": "doc"},
            source_ref={"filename": path.name},
        )

    raise RuntimeError(
        f"Cannot convert '{path.name}': install 'libreoffice' (headless) or 'antiword'."
    )
