"""Scanned PDF / images -> OCR text (German + English). Needs system tesseract + poppler."""
from __future__ import annotations

import logging
from pathlib import Path

from ..types import ParsedDoc

log = logging.getLogger(__name__)

_LOW_CONFIDENCE_THRESHOLD = 60  # mean tesseract word confidence below this -> flagged


def _ocr_image(img: "PIL.Image.Image", langs: str) -> tuple[str, bool]:  # noqa: F821
    """Return (text, is_low_confidence) for one image."""
    import pytesseract

    text = pytesseract.image_to_string(img, lang=langs)
    data = pytesseract.image_to_data(img, lang=langs, output_type=pytesseract.Output.DICT)
    confidences = [c for c in data["conf"] if isinstance(c, (int, float)) and c != -1]
    mean_conf = sum(confidences) / len(confidences) if confidences else 100
    return text, mean_conf < _LOW_CONFIDENCE_THRESHOLD


def parse(path: str | Path, langs: str = "deu+eng") -> ParsedDoc:
    """OCR each page (PDF) or the image directly. Flag low-confidence pages in metadata.

    For PDFs, requires poppler (pdf2image backend).
    For images (.png/.jpg/.jpeg), opens directly with Pillow.
    Returns ParsedDoc with text=joined pages and metadata including low_confidence_pages list.
    """
    from PIL import Image

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from pdf2image import convert_from_path

        images = convert_from_path(path)
    else:
        images = [Image.open(path)]

    pages_text: list[str] = []
    low_confidence_pages: list[int] = []

    for i, img in enumerate(images):
        text, is_low_conf = _ocr_image(img, langs)
        pages_text.append(text)
        if is_low_conf:
            low_confidence_pages.append(i)
            log.warning("ocr: low confidence on page %d of %s", i, path.name)

    return ParsedDoc(
        text="\n".join(pages_text),
        metadata={
            "filename": path.name,
            "type": "ocr",
            "pages": len(images),
            "low_confidence_pages": low_confidence_pages,
        },
        source_ref={"filename": path.name, "page_count": len(images)},
    )
