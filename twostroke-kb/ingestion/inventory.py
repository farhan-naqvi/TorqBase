"""Metadata-only inventory scanner for large local corpora.

This module deliberately does not parse, chunk, embed, or build KG facts. It
only catalogs filesystem metadata so a large corpus can be filtered before
selective ingestion.
"""
from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .format_router import EXT_MAP


TOPIC_FOLDERS = [
    "CAD",
    "Verbrennungsmotoren",
    "Oberflachenbehandlung",
    "Oberflächenbehandlung",
    "Elektrotechnik",
    "Aluminiumguss",
    "Propeller",
    "Konstruktionslehre",
    "Werkstoffkunde",
    "Luftfahrt",
    "Sonst. Stoffe",
    "Normen DIN ISO VDI FAR ASTM LURS",
    "Relevante Hirth-Information _ alt",
    "Vorlagen Testprotokolle",
    "Bauteilsicherheit und -zuverlaellisgkeit",
    "Bauteilsicherheit und -zuverlässigkeit",
    "Schulungen",
    "Bachelor_Master_Diplom_Doktorarbeiten",
    "Vibrationen",
    "Drehmomente",
    "Feinstellung-Zweitaktmotor",
]

DOCUMENT_EXTS = {".pdf", ".docx", ".doc", ".txt", ".md", ".html"}
PRESENTATION_EXTS = {".pptx", ".ppt"}
SPREADSHEET_EXTS = {".xlsx", ".xls", ".csv"}
CAD_EXTS = {
    ".step", ".stp", ".stl", ".dwg", ".dxf", ".iges", ".igs",
    ".sldprt", ".sldasm", ".catpart", ".catproduct",
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
ARCHIVE_EXTS = {".zip", ".rar", ".7z"}
SYSTEM_DIRS = {
    "$recycle.bin", ".git", ".hg", ".svn", "__pycache__", "node_modules",
    "system volume information", "thumbs.db",
}


@dataclass
class InventoryItem:
    batch_id: str
    root_path: str
    relative_path: str
    absolute_path: str
    file_name: str
    extension: str
    size_bytes: int
    modified_at: str | None
    topic: str
    category: str
    parser_name: str | None
    supported: bool
    directly_supported: bool
    metadata_only: bool
    needs_converter: bool
    skipped_reason: str | None
    status: str = "discovered"
    error: str | None = None


def classify_extension(filename_or_ext: str) -> dict[str, Any]:
    """Classify a file extension without opening the file."""
    suffix = filename_or_ext.lower()
    if not suffix.startswith("."):
        suffix = Path(suffix).suffix.lower()
    parser = EXT_MAP.get(suffix)

    if suffix in DOCUMENT_EXTS:
        category = "documents"
    elif suffix in PRESENTATION_EXTS:
        category = "presentations"
    elif suffix in SPREADSHEET_EXTS:
        category = "spreadsheets"
    elif suffix in CAD_EXTS:
        category = "cad_engineering"
    elif suffix in IMAGE_EXTS:
        category = "images"
    elif suffix in ARCHIVE_EXTS:
        category = "archives"
    else:
        category = "unknown"

    directly_supported = parser is not None
    needs_converter = suffix in {".doc", ".ppt", ".xls"}
    metadata_only = not directly_supported
    skipped_reason = None
    if not directly_supported:
        skipped_reason = (
            "metadata_only_cad" if category == "cad_engineering"
            else "metadata_only_archive" if category == "archives"
            else "unsupported_extension"
        )

    return {
        "extension": suffix,
        "category": category,
        "parser_name": parser,
        "supported": directly_supported,
        "directly_supported": directly_supported,
        "metadata_only": metadata_only,
        "needs_converter": needs_converter,
        "skipped_reason": skipped_reason,
    }


def scan_inventory(root_path: str | Path, max_files: int = 50000, batch_id: str | None = None) -> dict[str, Any]:
    """Scan a folder recursively and return/store metadata-only inventory rows."""
    started = time.perf_counter()
    root = Path(root_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Inventory root must be an existing directory: {root}")

    max_files = max(1, int(max_files or 1))
    batch_id = batch_id or f"inv-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    items: list[InventoryItem] = []
    errors: list[dict[str, str]] = []
    skipped_dirs = 0
    limit_reached = False

    def onerror(exc: OSError) -> None:
        errors.append({"path": getattr(exc, "filename", "") or "", "error": str(exc)})

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False, onerror=onerror):
        before = len(dirnames)
        dirnames[:] = [
            d for d in dirnames
            if not _should_skip_dir(Path(dirpath) / d)
        ]
        skipped_dirs += before - len(dirnames)

        for filename in filenames:
            if len(items) >= max_files:
                limit_reached = True
                break
            path = Path(dirpath) / filename
            try:
                if _is_hidden_or_system(path):
                    continue
                stat = path.stat()
                rel = path.relative_to(root)
                info = classify_extension(path.name)
                items.append(InventoryItem(
                    batch_id=batch_id,
                    root_path=str(root),
                    relative_path=str(rel),
                    absolute_path=str(path),
                    file_name=path.name,
                    extension=info["extension"],
                    size_bytes=int(stat.st_size),
                    modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    topic=_topic_for(rel),
                    category=info["category"],
                    parser_name=info["parser_name"],
                    supported=bool(info["supported"]),
                    directly_supported=bool(info["directly_supported"]),
                    metadata_only=bool(info["metadata_only"]),
                    needs_converter=bool(info["needs_converter"]),
                    skipped_reason=info["skipped_reason"],
                ))
            except OSError as exc:
                errors.append({"path": str(path), "error": str(exc)})
        if limit_reached:
            break

    store_error = None
    try:
        store_inventory_items(items)
    except Exception as exc:  # DB may not be running during local smoke tests.
        store_error = str(exc)

    elapsed = time.perf_counter() - started
    return {
        "batch_id": batch_id,
        "root_path": str(root),
        "items": [asdict(item) for item in items],
        "summary": summarize_items(items),
        "scan": {
            "scanned_files": len(items),
            "skipped_dirs": skipped_dirs,
            "unsupported_count": sum(1 for item in items if not item.supported),
            "errors_count": len(errors),
            "limit_reached": limit_reached,
            "max_files": max_files,
            "elapsed_seconds": round(elapsed, 3),
            "stored": store_error is None,
            "store_error": store_error,
        },
        "errors": errors[:50],
    }


def store_inventory_items(items: list[InventoryItem]) -> None:
    """Upsert inventory rows into Postgres."""
    if not items:
        return
    from config import get_connection

    conn = get_connection()
    try:
        _ensure_inventory_tables(conn)
        with conn.transaction():
            cur = conn.cursor()
            for item in items:
                cur.execute(
                    """
                    INSERT INTO file_inventory (
                        batch_id, root_path, relative_path, absolute_path, file_name,
                        extension, size_bytes, modified_at, topic, category,
                        parser_name, supported, directly_supported, metadata_only,
                        needs_converter, skipped_reason, status, error
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    ON CONFLICT (batch_id, relative_path) DO UPDATE SET
                        absolute_path = EXCLUDED.absolute_path,
                        file_name = EXCLUDED.file_name,
                        extension = EXCLUDED.extension,
                        size_bytes = EXCLUDED.size_bytes,
                        modified_at = EXCLUDED.modified_at,
                        topic = EXCLUDED.topic,
                        category = EXCLUDED.category,
                        parser_name = EXCLUDED.parser_name,
                        supported = EXCLUDED.supported,
                        directly_supported = EXCLUDED.directly_supported,
                        metadata_only = EXCLUDED.metadata_only,
                        needs_converter = EXCLUDED.needs_converter,
                        skipped_reason = EXCLUDED.skipped_reason,
                        status = EXCLUDED.status,
                        error = EXCLUDED.error
                    """,
                    (
                        item.batch_id, item.root_path, item.relative_path, item.absolute_path,
                        item.file_name, item.extension, item.size_bytes, item.modified_at,
                        item.topic, item.category, item.parser_name, item.supported,
                        item.directly_supported, item.metadata_only, item.needs_converter,
                        item.skipped_reason, item.status, item.error,
                    ),
                )
    finally:
        conn.close()


def list_inventory(batch_id: str | None = None, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
    from config import get_connection

    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))
    conn = get_connection()
    try:
        _ensure_inventory_tables(conn)
        cur = conn.cursor()
        if batch_id:
            cur.execute(
                """
                SELECT id, batch_id, root_path, relative_path, absolute_path, file_name,
                       extension, size_bytes, modified_at, topic, category, parser_name,
                       supported, directly_supported, metadata_only, needs_converter,
                       skipped_reason, status, error, created_at
                FROM file_inventory
                WHERE batch_id = %s
                ORDER BY id
                LIMIT %s OFFSET %s
                """,
                (batch_id, limit, offset),
            )
        else:
            cur.execute(
                """
                SELECT id, batch_id, root_path, relative_path, absolute_path, file_name,
                       extension, size_bytes, modified_at, topic, category, parser_name,
                       supported, directly_supported, metadata_only, needs_converter,
                       skipped_reason, status, error, created_at
                FROM file_inventory
                ORDER BY id DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
        items = [_row_to_dict(row) for row in cur.fetchall()]
        doc_ids = {_chunk_doc_id(item.get("file_name", "")) for item in items if item.get("file_name")}
        chunk_counts: dict[str, int] = {}
        if doc_ids:
            try:
                cur.execute(
                    """
                    SELECT doc_id, COUNT(*)
                    FROM chunks
                    WHERE doc_id = ANY(%s)
                    GROUP BY doc_id
                    """,
                    (list(doc_ids),),
                )
                chunk_counts = {row[0]: int(row[1] or 0) for row in cur.fetchall()}
            except Exception:
                chunk_counts = {}
        for item in items:
            item["chunk_count"] = chunk_counts.get(_chunk_doc_id(item.get("file_name", "")), 0)
        return items
    finally:
        conn.close()


def inventory_summary(batch_id: str | None = None) -> dict[str, Any]:
    """Return DB-backed inventory rollups."""
    from config import get_connection

    conn = get_connection()
    try:
        _ensure_inventory_tables(conn)
        cur = conn.cursor()
        where = "WHERE batch_id = %s" if batch_id else ""
        params = (batch_id,) if batch_id else ()

        cur.execute(f"SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM file_inventory {where}", params)
        total_files, total_size = cur.fetchone()

        cur.execute(
            f"""
            SELECT topic, COUNT(*), COALESCE(SUM(size_bytes), 0),
                   SUM(CASE WHEN supported THEN 1 ELSE 0 END),
                   SUM(CASE WHEN NOT supported THEN 1 ELSE 0 END)
            FROM file_inventory {where}
            GROUP BY topic
            ORDER BY COUNT(*) DESC, topic
            """,
            params,
        )
        by_topic = [
            {
                "topic": row[0] or "Unsorted",
                "count": int(row[1] or 0),
                "size_bytes": int(row[2] or 0),
                "supported": int(row[3] or 0),
                "unsupported": int(row[4] or 0),
            }
            for row in cur.fetchall()
        ]

        cur.execute(
            f"""
            SELECT category, COUNT(*), COALESCE(SUM(size_bytes), 0)
            FROM file_inventory {where}
            GROUP BY category
            ORDER BY COUNT(*) DESC, category
            """,
            params,
        )
        by_category = [
            {"category": row[0] or "unknown", "count": int(row[1] or 0), "size_bytes": int(row[2] or 0)}
            for row in cur.fetchall()
        ]

        cur.execute(
            f"""
            SELECT extension, COUNT(*), COALESCE(SUM(size_bytes), 0)
            FROM file_inventory {where}
            GROUP BY extension
            ORDER BY COUNT(*) DESC, extension
            LIMIT 40
            """,
            params,
        )
        by_extension = [
            {"extension": row[0] or "", "count": int(row[1] or 0), "size_bytes": int(row[2] or 0)}
            for row in cur.fetchall()
        ]

        cur.execute(
            f"""
            SELECT status, COUNT(*)
            FROM file_inventory {where}
            GROUP BY status
            ORDER BY COUNT(*) DESC, status
            """,
            params,
        )
        by_status = [{"status": row[0] or "unknown", "count": int(row[1] or 0)} for row in cur.fetchall()]

        return {
            "batch_id": batch_id,
            "total_files": int(total_files or 0),
            "total_size_bytes": int(total_size or 0),
            "by_topic": by_topic,
            "by_category": by_category,
            "by_extension": by_extension,
            "by_status": by_status,
        }
    finally:
        conn.close()


def dry_run_selected(
    topic: str | None = None,
    extensions: list[str] | None = None,
    inventory_ids: list[int] | None = None,
    max_files: int = 25,
    max_file_size_mb: int = 50,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Return what a controlled batch ingestion would do, without parsing files."""
    selected, skipped = _select_inventory_rows(
        topic=topic,
        extensions=extensions,
        inventory_ids=inventory_ids,
        max_files=max_files,
        max_file_size_mb=max_file_size_mb,
        skip_existing=skip_existing,
    )
    return _selection_report(selected, skipped, dry_run=True)


def ingest_selected(
    topic: str | None = None,
    extensions: list[str] | None = None,
    inventory_ids: list[int] | None = None,
    max_files: int = 25,
    max_file_size_mb: int = 50,
    skip_existing: bool = True,
    kg_enabled: bool = False,
    kg_max_chunks_per_doc: int = 20,
) -> dict[str, Any]:
    """Ingest selected supported inventory rows sequentially with per-file status."""
    from config import get_connection
    from ingestion.orchestrator import run_ingestion

    selected, skipped = _select_inventory_rows(
        topic=topic,
        extensions=extensions,
        inventory_ids=inventory_ids,
        max_files=max_files,
        max_file_size_mb=max_file_size_mb,
        skip_existing=skip_existing,
    )
    job_id = f"job-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    batch_id = selected[0].get("batch_id") if selected else None
    completed = failed = skipped_count = 0
    current_file = None
    last_error = None
    results: list[dict[str, Any]] = []

    conn = get_connection()
    try:
        _ensure_inventory_tables(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ingestion_jobs (job_id, batch_id, status, total_items, done_items)
            VALUES (%s, %s, %s, %s, 0)
            """,
            (job_id, batch_id, "running", len(selected)),
        )
        for row in selected:
            cur.execute(
                """
                INSERT INTO ingestion_job_items (job_id, inventory_id, status)
                VALUES (%s, %s, 'selected')
                ON CONFLICT (job_id, inventory_id) DO NOTHING
                """,
                (job_id, row["id"]),
            )
        conn.commit()

        for row in selected:
            current_file = row["file_name"]
            try:
                _mark_inventory_status(cur, row["id"], "ingesting", None)
                _mark_job_item_status(cur, job_id, row["id"], "ingesting", None)
                conn.commit()

                result = run_ingestion(
                    row["absolute_path"],
                    extra_metadata={
                        "topic": row.get("topic"),
                        "relative_path": row.get("relative_path"),
                        "file_type": row.get("extension"),
                        "source_title": row.get("file_name"),
                        "inventory_id": row.get("id"),
                        "inventory_batch_id": row.get("batch_id"),
                    },
                    kg_enabled=kg_enabled,
                    kg_max_chunks_per_doc=kg_max_chunks_per_doc,
                )
                _mark_inventory_status(cur, row["id"], "ingested", None)
                _mark_job_item_status(cur, job_id, row["id"], "ingested", None)
                completed += 1
                results.append({
                    "inventory_id": row["id"],
                    "file_name": row["file_name"],
                    "status": "ingested",
                    "chunks": result.chunks,
                    "skipped_duplicates": result.skipped_duplicates,
                    "version": result.version,
                })
            except Exception as exc:
                failed += 1
                try:
                    from ingestion.orchestrator import format_ingestion_error

                    last_error = format_ingestion_error(exc)
                except Exception:
                    last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                _mark_inventory_status(cur, row["id"], "failed", last_error)
                _mark_job_item_status(cur, job_id, row["id"], "failed", last_error)
                results.append({
                    "inventory_id": row["id"],
                    "file_name": row["file_name"],
                    "status": "failed",
                    "error": last_error,
                })
            finally:
                cur.execute(
                    """
                    UPDATE ingestion_jobs
                    SET done_items = %s, error = %s, updated_at = now()
                    WHERE job_id = %s
                    """,
                    (completed + failed, last_error, job_id),
                )
                conn.commit()

        for row in skipped:
            if row.get("id"):
                _mark_inventory_status(cur, row["id"], "skipped", row.get("skip_reason"))
                skipped_count += 1
        cur.execute(
            """
            UPDATE ingestion_jobs
            SET status = %s, done_items = %s, error = %s, updated_at = now()
            WHERE job_id = %s
            """,
            ("completed", completed + failed, last_error, job_id),
        )
        conn.commit()
    finally:
        conn.close()

    report = _selection_report(selected, skipped, dry_run=False)
    report.update({
        "job_id": job_id,
        "status": "completed",
        "completed": completed,
        "failed": failed,
        "skipped": skipped_count + len([r for r in skipped if not r.get("id")]),
        "current_file": current_file,
        "last_error": last_error,
        "results": results,
    })
    return report


def get_ingestion_job(job_id: str) -> dict[str, Any]:
    from config import get_connection

    conn = get_connection()
    try:
        _ensure_inventory_tables(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT job_id, batch_id, status, total_items, done_items, error, created_at, updated_at
            FROM ingestion_jobs WHERE job_id = %s
            """,
            (job_id,),
        )
        row = cur.fetchone()
        if not row:
            return {"job_id": job_id, "status": "not_found", "items": []}
        cur.execute(
            """
            SELECT ji.inventory_id, ji.status, ji.error, fi.file_name, fi.relative_path, fi.topic
            FROM ingestion_job_items ji
            LEFT JOIN file_inventory fi ON fi.id = ji.inventory_id
            WHERE ji.job_id = %s
            ORDER BY ji.id
            """,
            (job_id,),
        )
        return {
            "job_id": row[0],
            "batch_id": row[1],
            "status": row[2],
            "total_items": row[3],
            "done_items": row[4],
            "error": row[5],
            "created_at": row[6].isoformat() if row[6] else None,
            "updated_at": row[7].isoformat() if row[7] else None,
            "items": [
                {
                    "inventory_id": item[0],
                    "status": item[1],
                    "error": item[2],
                    "file_name": item[3],
                    "relative_path": item[4],
                    "topic": item[5],
                }
                for item in cur.fetchall()
            ],
        }
    finally:
        conn.close()


def summarize_items(items: list[InventoryItem]) -> dict[str, Any]:
    by_topic: dict[str, dict[str, Any]] = {}
    by_category: dict[str, dict[str, Any]] = {}
    by_extension: dict[str, dict[str, Any]] = {}
    by_status: dict[str, dict[str, Any]] = {}

    for item in items:
        _rollup(by_topic, item.topic or "Unsorted", item)
        _rollup(by_category, item.category or "unknown", item)
        _rollup(by_extension, item.extension or "", item)
        _rollup(by_status, item.status or "unknown", item)

    return {
        "total_files": len(items),
        "total_size_bytes": sum(item.size_bytes for item in items),
        "supported_files": sum(1 for item in items if item.supported),
        "unsupported_files": sum(1 for item in items if not item.supported),
        "by_topic": list(by_topic.values()),
        "by_category": list(by_category.values()),
        "by_extension": list(by_extension.values()),
        "by_status": list(by_status.values()),
    }


def _rollup(target: dict[str, dict[str, Any]], key: str, item: InventoryItem) -> None:
    row = target.setdefault(key, {"key": key, "topic": key, "category": key, "extension": key, "status": key, "count": 0, "size_bytes": 0, "supported": 0, "unsupported": 0})
    row["count"] += 1
    row["size_bytes"] += item.size_bytes
    if item.supported:
        row["supported"] += 1
    else:
        row["unsupported"] += 1


def _topic_for(relative_path: Path) -> str:
    parts = list(relative_path.parts)
    if len(parts) <= 1:
        return "Unsorted"
    first = _strip_numeric_prefix(parts[0])
    folded_first = _fold(first)
    for topic in TOPIC_FOLDERS:
        if _fold(topic) == folded_first:
            return topic
    return first or "Unsorted"


def _strip_numeric_prefix(name: str) -> str:
    cleaned = name.strip()
    while cleaned and (cleaned[0].isdigit() or cleaned[0] in ".-_ "):
        cleaned = cleaned[1:].strip()
    return cleaned or name.strip()


def _should_skip_dir(path: Path) -> bool:
    return _is_hidden_or_system(path) or path.is_symlink()


def _is_hidden_or_system(path: Path) -> bool:
    name = path.name
    lower = name.lower()
    return lower in SYSTEM_DIRS or lower.startswith(".") or lower.startswith("~$")


def _fold(value: str) -> str:
    return (
        value.casefold()
        .replace("ä", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ß", "ss")
    )


def _select_inventory_rows(
    topic: str | None,
    extensions: list[str] | None,
    inventory_ids: list[int] | None,
    max_files: int,
    max_file_size_mb: int,
    skip_existing: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from config import get_connection

    max_files = max(1, min(int(max_files or 25), 500))
    max_bytes = max(1, int(max_file_size_mb or 50)) * 1024 * 1024
    ext_set = {_normalize_ext(ext) for ext in (extensions or []) if _normalize_ext(ext)}
    id_set = {int(i) for i in (inventory_ids or []) if str(i).strip()}

    conn = get_connection()
    try:
        _ensure_inventory_tables(conn)
        cur = conn.cursor()
        clauses: list[str] = []
        params: list[Any] = []
        if topic:
            clauses.append("topic = %s")
            params.append(topic)
        if id_set:
            clauses.append("id = ANY(%s)")
            params.append(list(id_set))
        if ext_set:
            clauses.append("extension = ANY(%s)")
            params.append(sorted(ext_set))
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        cur.execute(
            f"""
            SELECT id, batch_id, root_path, relative_path, absolute_path, file_name,
                   extension, size_bytes, modified_at, topic, category, parser_name,
                   supported, directly_supported, metadata_only, needs_converter,
                   skipped_reason, status, error, created_at
            FROM file_inventory
            {where}
            ORDER BY topic, relative_path, id
            LIMIT %s
            """,
            (*params, max_files * 5),
        )
        rows = [_row_to_dict(row) for row in cur.fetchall()]

        selected: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for row in rows:
            reason = None
            if not row.get("supported"):
                reason = row.get("skipped_reason") or "unsupported_extension"
            elif int(row.get("size_bytes") or 0) > max_bytes:
                reason = "over_max_file_size"
            elif skip_existing and (row.get("status") == "ingested" or _already_ingested(cur, row)):
                reason = "already_ingested"
            elif row.get("absolute_path") and not Path(str(row["absolute_path"])).exists():
                reason = "missing_file"

            if reason:
                skipped.append({**row, "skip_reason": reason})
                continue
            if len(selected) < max_files:
                selected.append(row)
            else:
                skipped.append({**row, "skip_reason": "over_max_files_limit"})

        return selected, skipped
    finally:
        conn.close()


def _selection_report(selected: list[dict[str, Any]], skipped: list[dict[str, Any]], dry_run: bool) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    for row in skipped:
        reason = row.get("skip_reason") or "skipped"
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "dry_run": dry_run,
        "selected_count": len(selected),
        "total_size_bytes": sum(int(row.get("size_bytes") or 0) for row in selected),
        "supported_files": len(selected),
        "skipped_count": len(skipped),
        "skip_reasons": reasons,
        "files": [
            {
                "id": row.get("id"),
                "file_name": row.get("file_name"),
                "relative_path": row.get("relative_path"),
                "topic": row.get("topic"),
                "extension": row.get("extension"),
                "size_bytes": row.get("size_bytes"),
            }
            for row in selected[:100]
        ],
        "skipped_files": [
            {
                "id": row.get("id"),
                "file_name": row.get("file_name"),
                "relative_path": row.get("relative_path"),
                "reason": row.get("skip_reason"),
            }
            for row in skipped[:100]
        ],
    }


def _already_ingested(cur: Any, row: dict[str, Any]) -> bool:
    cur.execute(
        """
        SELECT 1 FROM documents
        WHERE storage_uri = %s
        LIMIT 1
        """,
        (row.get("absolute_path"),),
    )
    return cur.fetchone() is not None


def _mark_inventory_status(cur: Any, inventory_id: int, status: str, error: str | None) -> None:
    cur.execute(
        "UPDATE file_inventory SET status = %s, error = %s WHERE id = %s",
        (status, error, inventory_id),
    )


def _mark_job_item_status(cur: Any, job_id: str, inventory_id: int, status: str, error: str | None) -> None:
    cur.execute(
        """
        UPDATE ingestion_job_items
        SET status = %s, error = %s, updated_at = now()
        WHERE job_id = %s AND inventory_id = %s
        """,
        (status, error, job_id, inventory_id),
    )


def _normalize_ext(ext: str) -> str:
    value = str(ext or "").strip().lower()
    if not value:
        return ""
    return value if value.startswith(".") else f".{value}"


def _chunk_doc_id(filename: str) -> str:
    return re.sub(r"[^a-z0-9_.]", "_", str(filename or "unknown").lower())


def _row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    keys = [
        "id", "batch_id", "root_path", "relative_path", "absolute_path", "file_name",
        "extension", "size_bytes", "modified_at", "topic", "category", "parser_name",
        "supported", "directly_supported", "metadata_only", "needs_converter",
        "skipped_reason", "status", "error", "created_at",
    ]
    result = dict(zip(keys, row))
    for key in ("modified_at", "created_at"):
        if result.get(key) is not None:
            result[key] = result[key].isoformat()
    return result


def _ensure_inventory_tables(conn: Any) -> None:
    """Create inventory tables if the local DB has not applied schema.sql yet."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS file_inventory (
            id BIGSERIAL PRIMARY KEY,
            batch_id TEXT NOT NULL,
            root_path TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            absolute_path TEXT,
            file_name TEXT NOT NULL,
            extension TEXT,
            size_bytes BIGINT NOT NULL DEFAULT 0,
            modified_at TIMESTAMPTZ,
            topic TEXT,
            category TEXT NOT NULL DEFAULT 'unknown',
            parser_name TEXT,
            supported BOOLEAN NOT NULL DEFAULT false,
            directly_supported BOOLEAN NOT NULL DEFAULT false,
            metadata_only BOOLEAN NOT NULL DEFAULT false,
            needs_converter BOOLEAN NOT NULL DEFAULT false,
            skipped_reason TEXT,
            status TEXT NOT NULL DEFAULT 'discovered',
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (batch_id, relative_path)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS file_inventory_batch_idx ON file_inventory (batch_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS file_inventory_topic_idx ON file_inventory (topic)")
    cur.execute("CREATE INDEX IF NOT EXISTS file_inventory_status_idx ON file_inventory (status)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_jobs (
            id BIGSERIAL PRIMARY KEY,
            job_id TEXT UNIQUE NOT NULL,
            batch_id TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            total_items INT NOT NULL DEFAULT 0,
            done_items INT NOT NULL DEFAULT 0,
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_job_items (
            id BIGSERIAL PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES ingestion_jobs(job_id) ON DELETE CASCADE,
            inventory_id BIGINT REFERENCES file_inventory(id),
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (job_id, inventory_id)
        )
        """
    )
    conn.commit()
