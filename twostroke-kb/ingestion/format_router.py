"""Detect file type + language, dispatch to the right parser.

The router is the single entry point of GRAPH 1. Add new formats here.
"""
from __future__ import annotations

from pathlib import Path

from .types import ParsedDoc

# extension -> handler name (see parsers/)
EXT_MAP: dict[str, str] = {
    ".pdf": "pdf_or_ocr",      # decide digital vs scanned inside
    ".docx": "docx",
    ".doc": "doc_legacy",
    ".xlsx": "sheet",
    ".csv": "sheet",
    ".pptx": "pptx",
    ".txt": "text",
    ".md": "text",
    ".html": "text",
    ".png": "ocr",
    ".jpg": "ocr",
    ".jpeg": "ocr",
    ".url": "link",
}


def detect_language(text: str) -> str:
    """Return ISO code ('de' | 'en' | ...)."""
    try:
        from langdetect import detect

        return detect(text) if text.strip() else "unknown"
    except Exception:
        return "unknown"


def route(path: str | Path) -> ParsedDoc:
    """Dispatch a single uploaded file to its parser and return a ParsedDoc.

    Sets metadata['lang'] via langdetect after parsing.
    PDF figures are extracted and captioned when ENABLE_FIGURE_HANDLER=true.
    """
    import logging

    log = logging.getLogger(__name__)

    path = Path(path)
    handler = EXT_MAP.get(path.suffix.lower())
    if handler is None:
        raise ValueError(f"Unsupported format: {path.suffix}")

    if handler == "pdf_or_ocr":
        from .parsers import pdf_parser, ocr as ocr_parser

        doc = pdf_parser.parse(path)
        if not doc.text.strip():
            log.info("route: %s has no extractable text; falling back to OCR", path.name)
            doc = ocr_parser.parse(path)

        # STRETCH: merge figure captions (gated by ENABLE_FIGURE_HANDLER)
        from config import get_settings

        if get_settings().enable_figure_handler:
            try:
                from .parsers import figure_handler

                fig_doc = figure_handler.extract_and_caption(path)
                if fig_doc.text.strip():
                    doc = ParsedDoc(
                        text=doc.text + "\n\n" + fig_doc.text,
                        tables=doc.tables,
                        images=fig_doc.images,
                        metadata=doc.metadata,
                        source_ref=doc.source_ref,
                    )
            except Exception:
                log.warning("route: figure_handler failed for %s; continuing without figures", path.name)

    elif handler == "sheet":
        from .parsers import sheet_parser

        doc = sheet_parser.parse(path)
    elif handler == "ocr":
        from config import get_settings as _gs
        if _gs().enable_figure_handler:
            from .parsers import figure_handler
            doc = figure_handler.caption_image_file(path)
            if not doc.text.strip():
                # vision LLM returned nothing — fall back to Tesseract
                log.info("route: vision LLM empty for %s; falling back to OCR", path.name)
                from .parsers import ocr as ocr_parser
                doc = ocr_parser.parse(path)
        else:
            from .parsers import ocr as ocr_parser
            doc = ocr_parser.parse(path)
    elif handler == "doc_legacy":
        from .parsers import doc_convert
        import shutil

        if not shutil.which("libreoffice") and not shutil.which("antiword"):
            log.warning(
                "route: skipping '%s' — neither libreoffice nor antiword found. "
                "Run: sudo apt-get install -y libreoffice-headless",
                path.name,
            )
            doc = ParsedDoc(
                text=f"[Could not parse {path.name}: libreoffice or antiword is not installed.]",
                metadata={"filename": path.name, "type": "doc", "parse_error": "missing_converter"},
                source_ref={"filename": path.name},
            )
        else:
            doc = doc_convert.parse(path)
    elif handler == "docx":
        from .parsers import docx_parser

        doc = docx_parser.parse(path)
    elif handler == "pptx":
        from .parsers import pptx_parser

        doc = pptx_parser.parse(path)
    elif handler == "text":
        doc = ParsedDoc(
            text=path.read_text(errors="replace"),
            metadata={"filename": path.name, "type": "text"},
            source_ref={"filename": path.name},
        )
    elif handler == "link":
        from .parsers import link_handler

        doc = link_handler.parse(path)
    else:
        raise NotImplementedError(f"Parser not yet implemented for handler '{handler}' ({path.suffix})")

    doc.metadata["lang"] = detect_language(doc.text)
    return doc
