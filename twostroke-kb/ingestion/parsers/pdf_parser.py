"""Digital PDF → text with per-page sentinels for page-number tracking.

Each page boundary is marked with [PAGE_BREAK:N] (1-based).
The chunker strips these sentinels and uses them to record which page
each chunk came from in source_refs[0]["page"].
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

from ..types import ParsedDoc
from .math_normalizer import normalize_math_unicode

log = logging.getLogger(__name__)
_MIN_IMAGE_BYTES = 2048
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_CAPTION_RE = re.compile(r"\b(Figure|Fig\.|Abb\.|Bild|Diagram|Schema)\b", re.IGNORECASE)


def _doc_id_slug(path: Path) -> str:
    return re.sub(r"[^a-z0-9_.]", "_", path.name.lower())


def _image_filename_slug(path: Path) -> str:
    return re.sub(r"[^a-z0-9_.-]", "_", path.stem.lower())


def _already_extracted(doc_id: str) -> bool:
    try:
        from config import get_connection

        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM document_images WHERE doc_id = %s LIMIT 1", (doc_id,))
            return cur.fetchone() is not None
        finally:
            conn.close()
    except Exception:
        return False


def _caption_from_text(page_text: str) -> str:
    for line in page_text.splitlines():
        line = line.strip()
        if line and _CAPTION_RE.search(line):
            return line[:300]
    return ""


def parse(path: str | Path) -> ParsedDoc:
    """Extract text per page, inserting [PAGE_BREAK:N] sentinels between pages.

    Returns ParsedDoc with text="" for scanned PDFs (caller falls back to OCR).
    The sentinel format is: \\n\\n[PAGE_BREAK:N]\\n\\n
    The chunker uses these markers to attach page numbers to each chunk.
    """
    path = Path(path)
    doc_id = _doc_id_slug(path)
    doc = fitz.open(path)
    pages_text: list[tuple[int, str]] = []
    parsed_pages = 0
    failed_pages: list[int] = []
    images: list[dict] = []
    skip_images = _already_extracted(doc_id)
    image_dir: Path | None = None

    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise ValueError(f"pdf_parser: {path.name} is encrypted and cannot be read.")
        _ = doc.metadata
    except ValueError:
        doc.close()
        raise
    except Exception as exc:
        doc.close()
        raise ValueError(f"pdf_parser: cannot read metadata from {path.name}: {exc}") from exc

    try:
        from config import get_settings

        max_pages = max(1, int(get_settings().pdf_max_pages))
    except Exception:
        max_pages = 500
    total_pages = len(doc)
    if total_pages > max_pages:
        log.warning(
            "pdf_parser: %s has %d pages, processing only first %d",
            path.name,
            total_pages,
            max_pages,
        )
    page_range = range(min(total_pages, max_pages))

    if not skip_images:
        try:
            from config import get_settings

            image_dir = Path(get_settings().image_store_path)
            image_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            log.warning("pdf_parser: could not create image store; skipping image extraction")
            skip_images = True

    for page_idx in page_range:
        page_num = page_idx + 1
        try:
            page = doc[page_idx]
            page_text = normalize_math_unicode(page.get_text("text", sort=True))
        except Exception as exc:
            failed_pages.append(page_num)
            log.warning("pdf_parser: page %d failed in %s: %s", page_num, path.name, exc)
            continue

        if page_text.strip():
            parsed_pages += 1
        pages_text.append((page_num, page_text))
        if skip_images or image_dir is None:
            continue
        try:
            for img_idx, img in enumerate(page.get_images(full=True), 1):
                xref = img[0]
                extracted = doc.extract_image(xref)
                raw = extracted.get("image", b"")
                if len(raw) < _MIN_IMAGE_BYTES or len(raw) > _MAX_IMAGE_BYTES:
                    continue
                ext = str(extracted.get("ext") or "png").lower()
                filename = f"{_image_filename_slug(path)}_page{page_num}_img{img_idx}.{ext}"
                out_path = image_dir / filename
                out_path.write_bytes(raw)
                images.append({
                    "filename": filename,
                    "path": str(out_path),
                    "page": page_num,
                    "index": img_idx,
                    "caption": _caption_from_text(page_text),
                    "url": f"/images/{filename}",
                })
        except Exception as exc:
            log.warning("pdf_parser: image extraction failed for %s page %s: %s", path.name, page_num, exc)
    doc.close()

    if failed_pages:
        log.warning(
            "pdf_parser: %d pages failed and were skipped in %s: %s",
            len(failed_pages),
            path.name,
            failed_pages[:30],
        )

    # Join pages with sentinels so chunker can track page numbers
    # Page 1 is implicit (no leading sentinel); sentinels mark where page N begins
    if pages_text:
        first_page, first_text = pages_text[0]
        parts: list[str] = []
        if first_page != 1:
            parts.append(f"\n\n[PAGE_BREAK:{first_page}]\n\n")
        parts.append(first_text)
        for page_num, page_text in pages_text[1:]:
            parts.append(f"\n\n[PAGE_BREAK:{page_num}]\n\n")
            parts.append(page_text)
        full_text = normalize_math_unicode("".join(parts))
    else:
        full_text = ""

    if not full_text.strip():
        log.warning("pdf_parser: no extractable text in %s (scanned?)", path.name)
        if failed_pages and parsed_pages == 0:
            raise ValueError(
                f"pdf_parser: no text extracted from {path.name}; "
                f"{len(failed_pages)} page(s) failed or were unreadable."
            )

    return ParsedDoc(
        text=full_text,
        metadata={
            "filename": path.name,
            "pages": total_pages,
            "processed_pages": len(pages_text),
            "failed_pages": failed_pages,
            "type": "pdf",
        },
        images=images,
        source_ref={"filename": path.name, "page_count": total_pages},
    )
