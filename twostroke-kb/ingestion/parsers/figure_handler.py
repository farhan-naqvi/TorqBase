"""STRETCH — figure & diagram intelligence.

Extract embedded figures from PDFs, caption them with a vision-LLM (searchable text),
and store captions as chunks for semantic retrieval.

Gated by ENABLE_FIGURE_HANDLER=true in .env.
Vision model: LLM_VISION_MODEL (default llama3.2:3b) via LLM_BASE_URL.
Requires: PyMuPDF (fitz), Pillow.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

from ..types import ParsedDoc

log = logging.getLogger(__name__)

_MIN_DIMENSION = 64    # skip tiny icons / decorations
_UPSCALE_TO    = 512   # upscale images smaller than this so the vision model can read them

_CAPTION_PROMPT = (
    "This is a technical diagram or figure from a two-stroke engine manual. "
    "Describe precisely what you see: component names, labels, measurements, arrows, "
    "callouts, and any numeric values. Focus on technical content only."
)


def extract_and_caption(path: str | Path) -> ParsedDoc:
    """Extract images from every PDF page and caption each with the vision LLM.

    Returns a ParsedDoc whose text is the joined figure captions (one per line),
    and whose images list holds the raw PNG bytes for each figure.
    When ENABLE_FIGURE_HANDLER is false, immediately returns an empty ParsedDoc.
    """
    from config import get_settings

    settings = get_settings()
    path = Path(path)

    if not settings.enable_figure_handler:
        return ParsedDoc(
            text="",
            metadata={"filename": path.name, "type": "figure"},
            source_ref={"filename": path.name},
        )

    import fitz  # PyMuPDF
    from llm import describe_image

    pdf = fitz.open(str(path))
    captions: list[str] = []
    image_bytes_list: list[bytes] = []

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        for img_index, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            png_bytes = _extract_png(pdf, xref, page_num)
            if png_bytes is None:
                continue

            try:
                caption = describe_image(png_bytes, prompt=_CAPTION_PROMPT)
                if caption.strip():
                    captions.append(f"[Figure p.{page_num + 1}/{img_index + 1}]: {caption}")
                    image_bytes_list.append(png_bytes)
            except Exception as exc:
                log.debug("figure_handler: vision LLM failed for xref=%d p.%d: %s", xref, page_num + 1, exc)

    pdf.close()

    return ParsedDoc(
        text="\n\n".join(captions),
        images=image_bytes_list,
        metadata={
            "filename": path.name,
            "type": "figure",
            "figure_count": len(captions),
        },
        source_ref={"filename": path.name},
    )


def caption_image_file(path: str | Path) -> ParsedDoc:
    """Caption a standalone image file (PNG/JPG) with the vision LLM.

    Called directly from format_router for image uploads when ENABLE_FIGURE_HANDLER=true.
    Falls back to empty ParsedDoc if the gate is off.
    """
    from config import get_settings

    settings = get_settings()
    path = Path(path)

    if not settings.enable_figure_handler:
        return ParsedDoc(
            text="",
            metadata={"filename": path.name, "type": "figure"},
            source_ref={"filename": path.name},
        )

    from llm import describe_image

    try:
        png_bytes = _normalise_to_png(path.read_bytes())
        if png_bytes is None:
            raise ValueError("Could not decode image")
        caption = describe_image(png_bytes, prompt=_CAPTION_PROMPT)
    except Exception as exc:
        log.warning("figure_handler: caption_image_file failed for %s: %s", path.name, exc)
        caption = ""

    return ParsedDoc(
        text=f"[Figure]: {caption}" if caption.strip() else "",
        images=[png_bytes] if caption.strip() else [],
        metadata={"filename": path.name, "type": "figure", "figure_count": 1 if caption.strip() else 0},
        source_ref={"filename": path.name},
    )


def _extract_png(pdf: "fitz.Document", xref: int, page_num: int) -> bytes | None:  # noqa: F821
    """Extract one image from the PDF by xref, normalise to PNG. Returns None to skip."""
    try:
        base_image = pdf.extract_image(xref)
        img_bytes = base_image["image"]
    except Exception:
        return None
    return _normalise_to_png(img_bytes)


def _normalise_to_png(img_bytes: bytes) -> bytes | None:
    """Convert raw image bytes to an RGB PNG, upscaling tiny images for the vision model."""
    try:
        from PIL import Image

        pil_img = Image.open(io.BytesIO(img_bytes))

        # Skip tiny decorative images
        w, h = pil_img.size
        if w < _MIN_DIMENSION or h < _MIN_DIMENSION:
            return None

        # Upscale very small images so the vision model can read text/labels
        if w < _UPSCALE_TO or h < _UPSCALE_TO:
            scale = _UPSCALE_TO / min(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        if pil_img.mode not in ("RGB", "L"):
            pil_img = pil_img.convert("RGB")

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None
