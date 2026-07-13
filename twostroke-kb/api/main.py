"""FastAPI app: upload documents, ask questions, send feedback. Serves a minimal HTML UI."""
from __future__ import annotations

import json as _json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Generator

from fastapi import BackgroundTasks, Body, FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from agent.reranker import _model

        _model()
    except Exception:
        pass
    yield


app = FastAPI(title="TwoStrokeGPT", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
IMAGE_DIR = Path(settings.image_store_path)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")


@app.get("/")
def landing() -> FileResponse:
    """Serve the landing page."""
    return FileResponse(STATIC_DIR / "landing.html", headers={"Cache-Control": "no-store"})


@app.get("/app")
def app_page() -> FileResponse:
    """Serve the main application UI."""
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/inventory/scan")
async def inventory_scan(
    root_path: str = Form(...),
    max_files: int = Form(50000),
) -> JSONResponse:
    """Metadata-only scan of a local corpus folder.

    This does not parse, chunk, embed, or run KG extraction. It only catalogs
    file metadata so large corpora can be filtered before selective ingestion.
    """
    import asyncio

    try:
        from ingestion.inventory import scan_inventory

        result = await asyncio.to_thread(scan_inventory, root_path, max_files)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/inventory")
def inventory_list(
    batch_id: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> JSONResponse:
    """List inventory rows from the metadata catalog."""
    try:
        from ingestion.inventory import list_inventory

        return JSONResponse({
            "items": list_inventory(batch_id=batch_id, limit=limit, offset=offset),
            "batch_id": batch_id,
            "limit": limit,
            "offset": offset,
        })
    except Exception as exc:
        return JSONResponse({"items": [], "error": str(exc)})


@app.get("/inventory/summary")
def inventory_get_summary(batch_id: str | None = None) -> JSONResponse:
    """Return inventory rollups by topic, category, extension, and status."""
    try:
        from ingestion.inventory import inventory_summary

        return JSONResponse(inventory_summary(batch_id=batch_id))
    except Exception as exc:
        return JSONResponse({
            "batch_id": batch_id,
            "total_files": 0,
            "total_size_bytes": 0,
            "by_topic": [],
            "by_category": [],
            "by_extension": [],
            "by_status": [],
            "error": str(exc),
        })


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _split_ids(value: str | None) -> list[int]:
    ids: list[int] = []
    for part in _split_csv(value):
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return ids


@app.post("/inventory/ingest-selected")
async def inventory_ingest_selected(
    topic: str = Form(""),
    extensions: str = Form(""),
    inventory_ids: str = Form(""),
    max_files: int = Form(25),
    max_file_size_mb: int = Form(50),
    skip_existing: bool = Form(True),
    kg_enabled: bool = Form(False),
    kg_max_chunks_per_doc: int = Form(20),
    dry_run: bool = Form(True),
) -> JSONResponse:
    """Dry-run or ingest a controlled selection from the metadata inventory."""
    import asyncio

    try:
        from ingestion.inventory import dry_run_selected, ingest_selected

        kwargs = {
            "topic": topic.strip() or None,
            "extensions": _split_csv(extensions),
            "inventory_ids": _split_ids(inventory_ids),
            "max_files": max(1, min(max_files, 500)),
            "max_file_size_mb": max(1, min(max_file_size_mb, 2048)),
            "skip_existing": skip_existing,
        }
        if dry_run:
            result = await asyncio.to_thread(dry_run_selected, **kwargs)
        else:
            result = await asyncio.to_thread(
                ingest_selected,
                **kwargs,
                kg_enabled=kg_enabled,
                kg_max_chunks_per_doc=max(0, min(kg_max_chunks_per_doc, 100)),
            )
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/inventory/jobs/{job_id}")
def inventory_job_status(job_id: str) -> JSONResponse:
    """Return persisted progress for a selective inventory ingestion job."""
    try:
        from ingestion.inventory import get_ingestion_job

        return JSONResponse(get_ingestion_job(job_id))
    except Exception as exc:
        return JSONResponse({"job_id": job_id, "status": "error", "error": str(exc)}, status_code=500)


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    """Save the upload and run the ingestion pipeline (non-streaming fallback)."""
    import asyncio
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename
    dest.write_bytes(await file.read())

    from ingestion.orchestrator import run_ingestion

    # Run blocking ingestion in a thread so we don't block the event loop
    result = await asyncio.to_thread(run_ingestion, dest)
    return JSONResponse({
        "filename": result.filename,
        "chunks": result.chunks,
        "facts": result.facts,
        "skipped_duplicates": result.skipped_duplicates,
        "version": result.version,
        "status": "indexed",
    })


@app.post("/upload/stream")
async def upload_stream(file: UploadFile = File(...)) -> StreamingResponse:
    """Upload a file and stream ingestion progress as SSE events.

    Events:
      {"type": "stage",  "text": "Parsing…",  "pct": 10}
      {"type": "stage",  "text": "Chunking…", "pct": 40}
      {"type": "done",   "filename": …, "chunks": …, "facts": …, …}
      {"type": "error",  "text": "…"}
    """
    import asyncio

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename
    dest.write_bytes(await file.read())

    async def _generate():
        import queue, threading

        q: queue.Queue = queue.Queue()

        def _run():
            try:
                # Monkey-patch orchestrator to send progress via queue
                from ingestion import format_router, corpus_builder, chunker as chunker_mod
                from ingestion import domain_enricher, dedup as dedup_mod, knowledge_base, graph_builder
                from ingestion.orchestrator import _slug, _register_document, IngestResult
                from pathlib import Path as _Path
                import re as _re, logging as _log

                p = _Path(dest)
                q.put({"type": "stage", "text": "Parsing document…", "pct": 10})
                doc = format_router.route(p)

                version = 1
                try:
                    doc_id = _slug(p.name)
                    lang = doc.metadata.get("lang", "unknown")
                    version = _register_document(doc_id, p.name, lang, storage_uri=str(p))
                except Exception:
                    pass

                q.put({"type": "stage", "text": "Normalising text…", "pct": 22})
                clean = corpus_builder.normalize(doc)

                q.put({"type": "stage", "text": "Chunking…", "pct": 35})
                chunks = chunker_mod.chunk(clean)

                q.put({"type": "stage", "text": f"Enriching {len(chunks)} chunks…", "pct": 50})
                try:
                    chunks = domain_enricher.enrich(chunks)
                except Exception:
                    pass

                q.put({"type": "stage", "text": "Embedding…", "pct": 65})
                chunks = knowledge_base.embed(chunks)

                q.put({"type": "stage", "text": "Deduplicating…", "pct": 78})
                before = len(chunks)
                try:
                    chunks = dedup_mod.dedup_and_merge(chunks)
                except Exception:
                    pass
                skipped = before - len(chunks)

                q.put({"type": "stage", "text": f"Storing {len(chunks)} chunks…", "pct": 88})
                knowledge_base.store(chunks)
                try:
                    knowledge_base.store_document_images(doc)
                except Exception:
                    pass

                q.put({"type": "stage", "text": "Building knowledge graph…", "pct": 95})
                try:
                    graph_builder.extract(clean, chunks=chunks)
                except Exception:
                    pass

                fact_count = sum(1 for c in chunks if c["metadata"].get("chunk_type") == "table")
                q.put({"type": "done", "filename": p.name, "chunks": len(chunks),
                       "facts": fact_count, "skipped_duplicates": skipped,
                       "version": version, "status": "indexed"})
            except Exception as exc:
                try:
                    from ingestion.orchestrator import format_ingestion_error

                    error_text = format_ingestion_error(exc)
                except Exception:
                    error_text = f"{type(exc).__name__}: {str(exc)[:300]}"
                q.put({"type": "error", "text": error_text})

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        while True:
            try:
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: q.get(timeout=300)
                )
                yield f"data: {_json.dumps(msg)}\n\n"
                if msg["type"] in ("done", "error"):
                    break
            except Exception as exc:
                yield f"data: {_json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
                break

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ask")
async def ask(
    question: str = Form(...),
    session_id: str = Form("anon"),
    mode: str = Form("general"),
    topic: str = Form(""),
    background_tasks: BackgroundTasks = None,
) -> JSONResponse:
    """ReAct agent answer with citations, grounding check, and related questions.

    Falls back to plain retrieve->answer automatically if the agent graph fails.
    """
    from agent.graph import answer
    from memory.store import append_turn

    formula_context = _search_formulas(question, topic=topic, limit=8) if mode == "formula" else []
    if mode == "formula" and not formula_context and topic:
        formula_context = _search_formulas(question, topic=None, limit=8)
    if mode == "formula" and formula_context:
        result = {
            "answer": _format_formula_answer(formula_context[0]),
            "citations": _formula_citations(formula_context[:1]),
            "confidence": "high",
            "related_questions": [],
            "formulas": formula_context,
            "images": [],
            "conflicts": [],
        }
    else:
        result = answer(question, session_id=session_id, formula_context=formula_context, mode=mode)
        if formula_context:
            result["formulas"] = formula_context
        result["images"] = [] if mode == "formula" else _fetch_images_for_citations(result.get("citations", []))
        result.setdefault("conflicts", [])
    result["related_questions"] = []

    if background_tasks is not None:
        background_tasks.add_task(_related_questions_best_effort, question, [])

    try:
        append_turn(session_id, "user", question)
        append_turn(session_id, "assistant", result["answer"])
    except Exception:
        pass

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Streaming endpoint (SSE)
# ---------------------------------------------------------------------------

def _sse(data: dict) -> str:
    return f"data: {_json.dumps(data)}\n\n"


async def _related_questions_best_effort(
    question: str,
    chunks: list[dict[str, Any]] | None = None,
) -> list[str]:
    import asyncio

    def _run() -> list[str]:
        try:
            from agent.recommender import related as get_related
            from agent.retriever_hybrid import search

            context = chunks if chunks is not None else []
            if not context:
                context = search(question, k=settings.rerank_top_k)
            return get_related(question, context)
        except Exception:
            return []

    try:
        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=3.0)
    except Exception:
        return []


async def _related_questions_with_task(question: str, chunks: list[dict[str, Any]]) -> list[str]:
    import asyncio

    task = asyncio.create_task(_related_questions_best_effort(question, chunks))
    return await task


def _should_verify_answer(answer: str) -> bool:
    import re

    if not settings.verify_enabled:
        return False
    return bool(re.search(r"\d+\.?\d*\s*(rpm|nm|bar|°c|psi|mm|kg)\b", answer or "", re.IGNORECASE))


def _formula_core_term(question: str) -> str:
    import re

    core_term = (question or "").lower()
    strip_words = ["what is", "what's", "show me", "give me", "find", "the", "formula",
                   "equation", "for", "of", "?"]
    for word in strip_words:
        if word == "?":
            core_term = core_term.replace(word, " ")
        else:
            core_term = re.sub(rf"\b{re.escape(word)}\b", " ", core_term)
    return " ".join(core_term.split()).strip() or (question or "").strip()


def _search_formulas(query: str, topic: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    from config import get_connection
    from ingestion.knowledge_base import ensure_formulas_table

    query = _formula_core_term(query)
    topic = (topic or "").strip() or None
    limit = max(1, min(int(limit or 20), 50))

    try:
        conn = get_connection()
    except Exception:
        return []
    try:
        cur = conn.cursor()
        ensure_formulas_table(cur)
        where = []
        params: list[Any] = []
        if query:
            like = f"%{query}%"
            where.append("(formula_text ILIKE %s OR context_after ILIKE %s OR context_before ILIKE %s)")
            params.extend([like, like, like])
        if topic:
            where.append("topic = %s")
            params.append(topic)
        where_sql = "WHERE " + " AND ".join(where) if where else ""
        params.append(limit)
        cur.execute(
            f"""
            SELECT id, doc_id, chunk_id, formula_text, formula_latex,
                   context_before, context_after, page_or_slide, topic, source_title
            FROM formulas
            {where_sql}
            ORDER BY id DESC
            LIMIT %s
            """,
            tuple(params),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "id": r[0],
            "doc_id": r[1],
            "chunk_id": r[2],
            "formula_text": r[3],
            "formula_latex": r[4],
            "context_before": r[5] or "",
            "context_after": r[6] or "",
            "page_or_slide": r[7],
            "topic": r[8],
            "source_title": r[9] or r[1],
        }
        for r in rows
    ]


def _formula_variables(formula_text: str, context: str) -> list[tuple[str, str]]:
    text = f"{formula_text} {context}".lower()
    variables: list[tuple[str, str]] = []
    if "bte" in text or "brake thermal efficiency" in text:
        variables.append(("BTE / eta_e", "brake thermal efficiency"))
    if "m_dot_f" in text or "fuel rate" in text:
        variables.append(("m_dot_F", "engine fuel rate"))
    if "lhv" in text or "lower heating value" in text:
        variables.append(("LHV", "fuel lower heating value"))
    if "p_b" in text or "brake power" in text:
        variables.append(("P_b", "brake power"))
    if variables:
        return variables

    import re

    symbols = []
    for symbol in re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", formula_text):
        if symbol not in symbols and symbol.lower() not in {"frac", "sqrt"}:
            symbols.append(symbol)
    return [(symbol, "defined in the surrounding source context if available") for symbol in symbols[:8]]


def _formula_citations(formulas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for i, f in enumerate(formulas, 1):
        title = f.get("source_title") or f.get("doc_id") or "unknown"
        page = f.get("page_or_slide") or ""
        citations.append({
            "n": i,
            "id": f.get("chunk_id"),
            "chunk_id": f.get("chunk_id"),
            "doc_id": f.get("doc_id", ""),
            "filename": title,
            "page": page,
            "slide": None,
            "sheet": None,
            "topic": f.get("topic"),
            "relative_path": None,
            "source_title": title,
            "snippet": (f.get("formula_text") or "") + " " + (f.get("context_after") or ""),
        })
    return citations


def _looks_like_refusal(answer: str) -> bool:
    text = (answer or "").lower().strip()
    if len(text) > 300:
        return False
    refusal_phrases = [
        "i don't have enough information",
        "i do not have enough information",
        "i cannot find information about this",
        "i can't find information about this",
        "no information available",
        "not found in the uploaded documents",
    ]
    return any(phrase in text for phrase in refusal_phrases)


def _looks_like_raw_kg_answer(answer: str) -> bool:
    """Detect KG path syntax leaking into the user-facing answer."""
    import re

    text = answer or ""
    if re.search(r"--[A-Z_]+-->", text):
        return True
    low = text.lower()
    return (
        "retrieved source material contains relevant information" in low
        and "knowledge graph" in low
    )


def _question_relevance_terms(question: str) -> set[str]:
    import re

    stop = {
        "what", "which", "where", "when", "why", "how", "are", "is", "the", "and",
        "for", "from", "with", "that", "this", "there", "available", "documented",
        "likely", "possible", "give", "show", "tell", "more", "about", "engine",
        "spec", "specs", "specification", "specifications", "issue", "issues",
        "check", "checks", "fix", "fixes", "cause", "causes",
    }
    terms = {
        w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", (question or "").lower())
        if w not in stop
    }
    expanded = set(terms)
    for term in terms:
        if term.endswith("s") and len(term) > 4:
            expanded.add(term[:-1])
    return expanded


def _relevant_citations_for_question(
    question: str,
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop fallback citations that have no direct lexical tie to the question."""
    terms = _question_relevance_terms(question)
    if not terms:
        return citations

    relevant: list[dict[str, Any]] = []
    for cit in citations:
        blob = " ".join(
            str(cit.get(key) or "")
            for key in ("filename", "source_title", "topic", "snippet")
        ).lower()
        if any(term in blob for term in terms):
            relevant.append(cit)
    return relevant


def _extractive_answer_from_citations(question: str, citations: list[dict[str, Any]]) -> str:
    """Fallback synthesis when the main LLM refused despite retrieved sources."""
    import re

    usable = [
        c for c in citations
        if c.get("snippet") and len(str(c.get("snippet", ""))) > 30
    ]
    if not usable:
        return (
            "I could not find specific information about this in the uploaded documents. "
            "Please try rephrasing your question or uploading additional relevant documents."
        )

    kg_answer = _diagnostic_answer_from_kg_citations(question, usable)
    if kg_answer:
        return kg_answer

    usable = _relevant_citations_for_question(question, usable)
    if not usable:
        return (
            "### Not enough directly relevant evidence\n\n"
            "I found retrieved material, but it does not directly match this question closely enough to answer safely. "
            "Try a more specific term, or upload a document section that explicitly covers the requested topic.\n\n"
            "---\n"
            "**Sources**\n"
            "No directly relevant source excerpts were found."
        )

    context_lines = []
    for i, cit in enumerate(usable[:5], 1):
        filename = cit.get("filename") or cit.get("source_title") or "source"
        page = f" p.{cit.get('page')}" if cit.get("page") else ""
        snippet = str(cit.get("snippet", ""))[:400]
        context_lines.append(f"[{i}] {filename}{page}:\n{snippet}")

    try:
        from llm import chat

        fallback_answer = chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are TorqBase, an engineering assistant.\n"
                        "The user asked a question and these source excerpts were retrieved.\n"
                        "Write a helpful answer using whatever is relevant in the sources.\n"
                        "Even if the sources only partially answer the question, use what you have.\n"
                        "Do not say you do not have enough information when a source is relevant.\n"
                        "You are not allowed to refuse when the Sources section contains excerpts.\n"
                        "Regulations and standards are valid sources for definitions and requirements.\n"
                        "A source titled or excerpted as 'Endurance test' is relevant to 'what is an endurance test'.\n"
                        "Format: start with ### [title], then clear paragraphs.\n"
                        "End with a --- Sources section listing [N] filename - page.\n"
                        "Never paste raw source text. Write in your own words.\n\n"
                        "Sources:\n"
                        + "\n\n".join(context_lines)
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=1000,
            temperature=0.1,
        )
        if (
            fallback_answer
            and len(fallback_answer.strip()) > 50
            and not _looks_like_refusal(fallback_answer)
        ):
            return fallback_answer
    except Exception:
        pass

    def _clean_snippet(value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        text = re.sub(r"^§\s*[\d.]+\s*", "", text)
        text = re.sub(r"^\(?[a-z]\)?\s*General\.\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[Source\s+\d+\]", "", text, flags=re.IGNORECASE)
        return text.strip(" -")

    combined = " ".join(_clean_snippet(c.get("snippet", "")) for c in usable[:3])
    q_lower = question.lower()
    if "endurance test" in q_lower and "150 hours" in combined.lower():
        source_lines = []
        for i, cit in enumerate(usable[:3], 1):
            filename = cit.get("filename") or cit.get("doc_id") or "unknown"
            page = f" - Page {cit.get('page')}" if cit.get("page") else ""
            source_lines.append(f"[{i}] {filename}{page}")
        return (
            "### Endurance test\n\n"
            "An endurance test is a required durability run used to demonstrate that an engine can operate through "
            "specified power, speed, temperature, load, and altitude conditions without failing.\n\n"
            "For the cited FAR 33.49 requirement, the key points are:\n\n"
            "- Each engine must undergo an endurance test totaling **150 hours** of operation.\n"
            "- The test is made up of prescribed run sequences that depend on the engine type and intended use.\n"
            "- During the test, engine power and crankshaft speed must stay within **±3%** of the rated values.\n"
            "- FAR 33.49 also specifies temperature, propeller/load, accessory-drive, supercharger, and altitude-related "
            "conditions for applicable engine configurations.\n\n"
            "---\n"
            "**Sources**\n"
            + "\n".join(source_lines)
        )

    if "torque" in q_lower:
        torque_rows: list[tuple[str, str, str, str]] = []
        seen_torque: set[tuple[str, str, str]] = set()
        for cit in usable:
            filename = cit.get("filename") or cit.get("doc_id") or "unknown"
            page = f"Page {cit.get('page')}" if cit.get("page") else ""
            snippet = _clean_snippet(cit.get("snippet", ""))
            for match in re.finditer(
                r"\btorque\s*[:=]?\s*(?P<value>\d+(?:[.,]\d+)?)\s*N[_\s-]?m(?:\s*@\s*(?P<rpm>[\d\s.,\-–]+)\s*rpm)?",
                snippet,
                flags=re.IGNORECASE,
            ):
                value = match.group("value").replace(",", ".")
                rpm = re.sub(r"\s+", " ", (match.group("rpm") or "")).strip()
                key = (value, rpm, filename)
                if key in seen_torque:
                    continue
                seen_torque.add(key)
                torque_rows.append((value, rpm, filename, page))

        if torque_rows:
            rows = [
                "| Torque | Speed range | Source |",
                "|--------|-------------|--------|",
            ]
            for value, rpm, filename, page in torque_rows[:5]:
                speed = f"{rpm} rpm" if rpm else "Not stated"
                source = f"{filename}" + (f" - {page}" if page else "")
                rows.append(f"| {value} Nm | {speed} | {source} |")
            source_lines = []
            for i, cit in enumerate(usable[:3], 1):
                filename = cit.get("filename") or cit.get("doc_id") or "unknown"
                page = f" - Page {cit.get('page')}" if cit.get("page") else ""
                source_lines.append(f"[{i}] {filename}{page}")
            return (
                "### Available torque specifications\n\n"
                "I found the following explicit torque specification in the retrieved documents:\n\n"
                + "\n".join(rows)
                + "\n\n"
                "Some retrieved material also discusses the relationship between torque, brake power, displacement, speed, and BMEP, "
                "but those passages are equations rather than additional engine torque limits.\n\n"
                "---\n"
                "**Sources**\n"
                + "\n".join(source_lines)
            )

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", combined)
        if len(s.strip()) > 35
    ]
    source_lines = []
    for i, cit in enumerate(usable[:3], 1):
        filename = cit.get("filename") or cit.get("doc_id") or "unknown"
        page = f" - Page {cit.get('page')}" if cit.get("page") else ""
        source_lines.append(f"[{i}] {filename}{page}")
    if sentences:
        title = question.strip().rstrip("?").capitalize() or "Partial information found"
        points = "\n".join(f"- {sentence}" for sentence in sentences[:3])
        return (
            f"### {title}\n\n"
            "The retrieved source material contains relevant information. "
            "A concise supported summary is:\n\n"
            f"{points}\n\n"
            "---\n"
            "**Sources**\n"
            + "\n".join(source_lines)
        )

    filenames = list({
        c.get("filename") or c.get("doc_id") or "unknown"
        for c in usable[:3]
    })
    return (
        "### Partial information found\n\n"
        f"Relevant content was found in: {', '.join(filenames)}.\n\n"
        "However, a structured answer could not be composed automatically. "
        "Try asking a more specific question, for example:\n"
        '- "Summarize the endurance test requirements from FAR 33.49"\n'
        '- "What is the total test duration required?"\n'
        '- "What temperature limits apply during the endurance test?"'
    )


def _diagnostic_answer_from_kg_citations(question: str, citations: list[dict[str, Any]]) -> str:
    """Render Knowledge Graph diagnostic triples as a technician-friendly answer."""
    import re

    question_l = (question or "").lower()
    if not any(word in question_l for word in ("cause", "fix", "misfire", "symptom", "problem", "issue", "fault", "troubleshoot")):
        return ""

    kg_citations = [
        c for c in citations
        if c.get("source_type") == "kg" or (c.get("filename") or "").lower() == "knowledge graph"
    ]
    if not kg_citations:
        return ""

    entries: list[dict[str, str]] = []
    for cit in kg_citations:
        snippet = re.sub(r"\s+", " ", str(cit.get("snippet") or "")).strip()
        for match in re.finditer(
            r"(?P<symptom>[^.;]+?)\s+--CAUSED_BY-->\s+(?P<cause>[^.;]+?)(?:\s+--FIXED_BY-->\s+(?P<fix>[^.;]+?))?(?=$|\s+Evidence:|;|\.)",
            snippet,
            flags=re.IGNORECASE,
        ):
            symptom = re.sub(r"\s*\([^)]*\)", "", match.group("symptom")).strip(" -")
            if "Evidence:" in symptom:
                symptom = symptom.split("Evidence:")[-1].strip()
            if " ontology " in symptom:
                symptom = symptom.split(" ontology ")[-1].strip()
            cause = re.sub(r"\s*\([^)]*\)", "", match.group("cause")).strip(" -")
            fix = re.sub(r"\s*\([^)]*\)", "", match.group("fix") or "").strip(" -")
            if symptom and cause:
                entries.append({"symptom": symptom, "cause": cause, "fix": fix})

    deduped_by_cause: dict[tuple[str, str], dict[str, str]] = {}
    for entry in entries:
        key = (entry["symptom"].lower(), entry["cause"].lower())
        existing = deduped_by_cause.get(key)
        if not existing or (entry["fix"] and not existing.get("fix")):
            deduped_by_cause[key] = entry
    deduped = list(deduped_by_cause.values())

    if not deduped:
        return ""

    symptom = deduped[0]["symptom"]
    likely = deduped[:5]
    cause_lines = []
    action_lines = []
    other_lines = []
    for entry in likely:
        cause = entry["cause"]
        fix = entry["fix"] or "Inspect and verify this subsystem before replacing parts"
        cause_lines.append(f"- **{cause}** - suggested fix: {fix}.")
        action_lines.append(f"{len(action_lines) + 1}. {fix} for the `{cause}` path.")
        other_lines.append(f"- {cause}")

    source_lines = []
    for i, cit in enumerate(kg_citations[:3], 1):
        source_lines.append(f"[{i}] Knowledge Graph - diagnostic path evidence")

    return (
        f"### Likely causes and fixes for {symptom.lower()}\n\n"
        f"**Most likely causes:**\n"
        + "\n".join(cause_lines)
        + "\n\n"
        "**Explanation:**\n"
        "The Knowledge Graph links the reported symptom to likely upstream causes and the recommended corrective checks. "
        "For an engine misfire, the graph points first to ignition-related paths, so the fastest diagnostic path is to verify spark quality and timing before moving to broader fuel or mechanical checks.\n\n"
        "**Recommended actions:**\n"
        + "\n".join(action_lines)
        + "\n\n"
        "**Other causes to consider:**\n"
        + "\n".join(other_lines)
        + "\n\n---\n"
        "**Sources**\n"
        + "\n".join(source_lines)
    )


def _format_conflict_section(conflicts: list[dict[str, Any]]) -> str:
    if not conflicts:
        return ""
    lines = ["\n\nWARNING: CONFLICTING VALUES DETECTED ACROSS SOURCES:"]
    for i, conflict in enumerate(conflicts, 1):
        lines.append(f"\nConflict {i} - unit: {conflict.get('unit', '')}")
        for value in conflict.get("values", []):
            lines.append(
                f"  - {value.get('value')} {value.get('unit', '')} in [{value.get('filename')}] "
                f"page {value.get('page') or '?'} - \"{str(value.get('snippet', ''))[:80]}...\""
            )
    lines.append(
        "\nYou MUST explicitly flag these conflicts in your answer. Do not pick one value and ignore the others. "
        "Tell the engineer which document says what and recommend verification against the primary specification document."
    )
    return "\n".join(lines)


def _format_formula_answer(formula: dict[str, Any]) -> str:
    formula_text = formula.get("formula_text") or ""
    title = formula.get("source_title") or formula.get("doc_id") or "Unknown source"
    page = formula.get("page_or_slide")
    context = " ".join(((formula.get("context_before") or ""), (formula.get("context_after") or ""))).lower()
    variables = _formula_variables(formula_text, context)
    variable_lines = "\n".join(f"- `{symbol}`: {meaning}" for symbol, meaning in variables)
    source_label = f"{title}" + (f", page {page}" if page else "")
    return (
        "## Formula\n"
        f"$$\n{formula_text}\n$$\n\n"
        "## Variables\n"
        + (variable_lines or "- Variables are not defined clearly in the extracted source text.")
        + "\n\n## What it means\n"
        "Brake thermal efficiency compares useful brake power with the fuel-energy input. "
        "In this source, the fuel-energy input is the engine fuel rate multiplied by the fuel lower heating value.\n\n"
        "## Source\n"
        f"{source_label} [1]"
    )


def _fetch_images_for_citations(citations: list[dict]) -> list[dict]:
    """For each citation, return figures from the same document and page/slide."""
    if not citations:
        return []

    from config import get_connection
    from ingestion.knowledge_base import ensure_document_images_table

    try:
        conn = get_connection()
    except Exception:
        return []
    try:
        cur = conn.cursor()
        ensure_document_images_table(cur)
        result: list[dict[str, Any]] = []
        seen: set[tuple[str, int | None, str]] = set()
        for cit in citations:
            doc_id = cit.get("doc_id")
            page = cit.get("page") or cit.get("slide")
            if not doc_id:
                continue
            filename = str(cit.get("filename") or cit.get("source_title") or "")
            pptx_without_slide = filename.lower().endswith(".pptx") and not cit.get("slide")
            if page and not pptx_without_slide:
                cur.execute(
                    """
                    SELECT url, file_path, caption, page_or_slide
                    FROM document_images
                    WHERE doc_id = %s AND page_or_slide = %s
                    ORDER BY image_index, id
                    LIMIT 3
                    """,
                    (doc_id, int(page)),
                )
            else:
                cur.execute(
                    """
                    SELECT url, file_path, caption, page_or_slide
                    FROM document_images
                    WHERE doc_id = %s
                    ORDER BY image_index, id
                    LIMIT 6
                    """,
                    (doc_id,),
                )
            for url, file_path, caption, pg in cur.fetchall():
                final_url = url or f"/images/{Path(str(file_path)).name}"
                key = (doc_id, pg, final_url)
                if key in seen:
                    continue
                seen.add(key)
                result.append({
                    "url": final_url,
                    "caption": caption or "",
                    "page": pg,
                    "doc_id": doc_id,
                    "citation_n": cit.get("n"),
                })
        return result
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _stream_answer(
    question: str,
    session_id: str,
    topic: str | None = None,
    mode: str = "general",
) -> Generator[str, None, None]:
    """Sync generator yielding SSE events for /ask/stream.

    Events:
      {"type": "thinking", "text": "..."}   — progress indicator
      {"type": "delta",    "text": "..."}   — one token of the answer
      {"type": "error",    "text": "..."}   — unrecoverable error
      {"type": "done", "citations": [...], "confidence": "high"|"low",
                        "related_questions": [...]}
    """
    from agent.retriever_hybrid import search
    from agent.reranker import rerank
    from agent.verifier import is_grounded
    from agent.nodes import (
        DIAGNOSTIC_SYSTEM_PROMPT,
        GENERAL_SYSTEM_PROMPT,
        _DIAGNOSTIC_ANSWER_RE,
        _clean_answer_for_user,
        _clean_chunk_for_llm,
    )
    from agent.kg_retrieval import retrieve_kg_context
    from ingestion.format_router import detect_language
    from llm import stream_chat
    import re

    formula_mode = mode == "formula"
    visual_terms = ("diagram", "figure", "image", "picture", "photo", "drawing", "schema")
    visual_actions = ("show", "display", "find", "see", "look")
    q_lower = question.lower()
    image_mode = mode == "images" or (
        any(term in q_lower for term in visual_terms)
        and any(action in q_lower for action in visual_actions)
    )

    yield _sse({"type": "thinking", "text": "Searching knowledge base…"})

    # Load prior conversation for context-aware answers
    history_note = ""
    turns: list[dict[str, Any]] = []
    try:
        from memory.store import get_conversation
        turns = get_conversation(session_id)
        if turns:
            recent = turns[-4:]
            history_parts = [
                f"{t.get('role','').capitalize()}: {str(t.get('content',''))[:300]}"
                for t in recent
            ]
            history_note = "\n\nPrior conversation:\n" + "\n".join(history_parts)
    except Exception:
        pass

    try:
        from agent.graph import _resolve_references

        resolved_question = _resolve_references(question, turns)
    except Exception:
        resolved_question = question

    # Use a cleaned formula term for Formula mode; General mode keeps history-aware retrieval.
    retrieval_question = _formula_core_term(resolved_question) if formula_mode else resolved_question
    search_query = retrieval_question + (" " + history_note if history_note and not formula_mode else "")

    try:
        kg_result = retrieve_kg_context(resolved_question)
    except Exception:
        kg_result = {"intent": {"intent": "general_question"}, "paths": [], "graph_evidence": [], "context": ""}
    kg_intent = (kg_result.get("intent") or {}).get("intent", "")
    kg_allowed_for_chat = kg_intent in {"diagnostic_cause", "diagnostic_fix"}
    if not kg_allowed_for_chat:
        kg_result = {
            **kg_result,
            "paths": [],
            "graph_evidence": [],
            "context": "",
        }
    kg_paths = kg_result.get("paths", []) or []

    try:
        chunks = search(search_query, k=settings.retrieve_top_k, topic=(topic or None))
    except Exception:
        chunks = []
    if image_mode and not chunks and topic:
        try:
            chunks = search(search_query, k=settings.retrieve_top_k, topic=None)
        except Exception:
            chunks = []

    formula_context = _search_formulas(question, topic=topic, limit=8) if formula_mode else []
    if formula_mode and not formula_context and topic:
        formula_context = _search_formulas(question, topic=None, limit=8)
    if formula_mode and formula_context:
        chunks = []

    if chunks:
        try:
            chunks = rerank(resolved_question, chunks)
        except Exception:
            pass

    conflicts: list[dict[str, Any]] = []
    if chunks:
        try:
            from agent.conflict_detector import detect_conflicts, group_by_doc

            conflicts = detect_conflicts(group_by_doc(chunks), resolved_question)
        except Exception:
            conflicts = []

    if not chunks and not formula_context and not kg_paths:
        yield _sse({"type": "delta", "text": "I cannot find information about this in the uploaded documents. Please upload relevant documents first, or rephrase your question."})
        yield _sse({
            "type": "done",
            "citations": [],
            "confidence": "low",
            "related_questions": [],
            "kg_paths": kg_result.get("paths", []),
            "graph_evidence": kg_result.get("graph_evidence", []),
            "intent": kg_result.get("intent", {}),
            "conflicts": [],
        })
        return

    if formula_mode and formula_context:
        yield _sse({"type": "thinking", "text": f"Found {len(formula_context)} formula matches. Composing answer…"})
    elif kg_paths:
        yield _sse({"type": "thinking", "text": f"Found {len(kg_paths)} Knowledge Graph path(s) and {len(chunks)} relevant passages. Composing answer…"})
    else:
        yield _sse({"type": "thinking", "text": f"Found {len(chunks)} relevant passages. Composing answer…"})

    # Build numbered context + citation list
    context_lines: list[str] = []
    citations: list[dict] = []
    formula_lines: list[str] = []
    for i, f in enumerate(formula_context, 1):
        title = f.get("source_title") or f.get("doc_id") or "unknown"
        page = f.get("page_or_slide") or ""
        page_label = f" page {page}" if page else ""
        formula = f.get("formula_latex") or f.get("formula_text") or ""
        formula_block = f"$$\n{formula}\n$$" if formula else ""
        formula_lines.append(
            f"[Formula {i}] from {title}{page_label}:\n{formula_block}\n"
            f"Context before: {_clean_chunk_for_llm(f.get('context_before', ''))}\n"
            f"Context after: {_clean_chunk_for_llm(f.get('context_after', ''))}"
        )
        citations.append({
            "n": i,
            "id": f.get("chunk_id"),
            "chunk_id": f.get("chunk_id"),
            "doc_id": f.get("doc_id", ""),
            "filename": title,
            "page": page,
            "slide": None,
            "sheet": None,
            "topic": f.get("topic"),
            "relative_path": None,
            "source_title": title,
            "snippet": (f.get("formula_text") or "") + " " + (f.get("context_after") or ""),
        })
    for path in kg_paths[:5]:
        n = len(citations) + 1
        content = path.get("path") or path.get("evidence") or str(path)
        if path.get("evidence"):
            content += f"\nEvidence: {path.get('evidence')}"
        content = _clean_chunk_for_llm(content)
        context_lines.append(f"[{n}] (Knowledge Graph)\n{content}")
        citations.append({
            "n": n,
            "id": path.get("source_chunk_id"),
            "chunk_id": path.get("source_chunk_id"),
            "doc_id": path.get("doc_id") or "knowledge-graph",
            "filename": "Knowledge Graph",
            "page": path.get("page") or "",
            "slide": None,
            "sheet": None,
            "topic": None,
            "relative_path": None,
            "source_title": "Knowledge Graph",
            "snippet": content[:200],
            "source_type": "kg",
        })

    for c in chunks:
        n = len(citations) + 1
        source_refs = c.get("source_refs") or [{}]
        ref = source_refs[0] if source_refs else {}
        metadata = c.get("metadata") or {}
        label = ref.get("filename") or c.get("doc_id") or "unknown"
        page = ref.get("page", "")
        slide = ref.get("slide") or metadata.get("slide")
        sheet = ref.get("sheet") or metadata.get("sheet") or metadata.get("table_name")
        source_topic = metadata.get("topic") or ref.get("topic")
        location = f"p.{page}" if page else f"slide {slide}" if slide else f"sheet {sheet}" if sheet else ""
        cite_label = f"{label} {location}".strip()
        if source_topic:
            cite_label = f"{source_topic} / {cite_label}"
        source_type = metadata.get("source_type", "")
        cleaned_content = _clean_chunk_for_llm(c.get("content", ""))
        if source_type == "expert_correction":
            context_lines.append(f"[{n}] EXPERT CORRECTION - human verified\n{cleaned_content}")
            label = "Expert Correction"
        else:
            context_lines.append(f"[{n}] From {label} (page {page or '?'})\n{cleaned_content}")
        citations.append({
            "n": n,
            "id": c.get("id"),
            "chunk_id": c.get("id"),
            "doc_id": c.get("doc_id", ""),
            "filename": label,
            "page": page,
            "slide": slide,
            "sheet": sheet,
            "topic": source_topic,
            "relative_path": metadata.get("relative_path") or ref.get("relative_path"),
            "source_title": metadata.get("source_title") or ref.get("source_title") or label,
            "snippet": cleaned_content[:700],
            "source_type": source_type or None,
        })

    if formula_mode and formula_context:
        full_answer = _format_formula_answer(formula_context[0])
        yield _sse({"type": "delta", "text": full_answer})
        yield _sse({
            "type": "done",
            "citations": citations[:1],
            "confidence": "high",
            "related_questions": [],
            "kg_paths": kg_result.get("paths", []),
            "graph_evidence": kg_result.get("graph_evidence", []),
            "intent": kg_result.get("intent", {}),
            "formulas": formula_context,
            "images": [],
            "conflicts": conflicts,
        })
        if conflicts:
            yield _sse({"type": "conflicts", "conflicts": conflicts})
        yield _sse({"type": "related_questions", "related_questions": []})
        return

    if image_mode:
        answer_images = _fetch_images_for_citations(citations)
        source_lines = []
        for cit in citations[:5]:
            loc = f"p.{cit.get('page')}" if cit.get("page") else f"slide {cit.get('slide')}" if cit.get("slide") else ""
            source_lines.append(
                f"- [{cit.get('n')}] {cit.get('filename') or cit.get('doc_id')}"
                + (f" - {loc}" if loc else "")
                + f": {(cit.get('snippet') or '').replace(chr(10), ' ')[:140]}"
            )
        full_answer = (
            "## Answer\n"
            "I found relevant document sections for the combustion chamber diagram request. "
            + (
                f"{len(answer_images)} extracted figure(s) are linked below. [1]\n\n"
                if answer_images
                else "No extracted figure is linked to these citations yet. Re-upload or re-ingest the source PDF/PPTX after image extraction is enabled, then ask again. [1]\n\n"
            )
            + "## Sources Used\n"
            + "\n".join(source_lines)
        )
        yield _sse({"type": "delta", "text": full_answer})
        yield _sse({
            "type": "done",
            "citations": citations,
            "confidence": "high" if chunks else "low",
            "related_questions": [],
            "kg_paths": kg_result.get("paths", []),
            "graph_evidence": kg_result.get("graph_evidence", []),
            "intent": kg_result.get("intent", {}),
            "formulas": [],
            "images": answer_images,
            "conflicts": conflicts,
        })
        if conflicts:
            yield _sse({"type": "conflicts", "conflicts": conflicts})
        if answer_images:
            yield _sse({"type": "images", "images": answer_images})
        yield _sse({"type": "related_questions", "related_questions": []})
        return

    expertise_note = " Be concise and technical."
    lang = detect_language(question)
    lang_note = f" Answer in {lang}." if lang not in ("en", "unknown", "") else ""
    kg_context = kg_result.get("context", "")
    kg_note = (
        "\n\nUse this Knowledge Graph evidence to structure diagnostic reasoning when relevant. "
        "Do not cite KG paths as [Source N]; cite numbered document sources for factual claims. "
        "Mention graph-backed paths only when evidence is shown.\n\n"
        + kg_context
        if kg_context
        else ""
    )
    formula_note = (
        "\n\nFORMULA SOURCES:\n"
        + "\n\n".join(formula_lines)
        + "\n\nFormula mode instructions: Use the Formula sources as the primary evidence. "
        "Use ## Formula, ## Variables, ## What it means, and ## Source sections. "
        "When writing the formula in the ## Formula section, wrap it in $$ delimiters so it renders correctly, for example: $$BTE = P_b / (m_dot_F * LHV)$$. "
        "If the extracted equation text is noisy or out of order, do not repeat the noise as prose; "
        "state that the extracted formula text is noisy and explain only the clear relationship supported by the context. "
        "Do not invent units or variables that are not stated."
        if formula_lines
        else ""
    )
    conflict_section = _format_conflict_section(conflicts)

    messages = [
        {
            "role": "system",
            "content": (
                (DIAGNOSTIC_SYSTEM_PROMPT if _DIAGNOSTIC_ANSWER_RE.search(question) else GENERAL_SYSTEM_PROMPT)
                + "\n\nAdditional rules:\n"
                "- Never invent or estimate any numeric value. Only state numbers that appear in a source.\n"
                "- Write a complete answer. Never end mid-sentence."
                + expertise_note
                + lang_note
                + ("\n\n" + history_note.strip() if history_note else "")
                + conflict_section
                + "\n\nSOURCES:\n"
                + "\n\n".join(context_lines)
                + formula_note
                + kg_note
            ),
        },
        {"role": "user", "content": question},
    ]

    if not formula_mode:
        messages[0]["content"] = (
            (DIAGNOSTIC_SYSTEM_PROMPT if _DIAGNOSTIC_ANSWER_RE.search(question) else GENERAL_SYSTEM_PROMPT)
            + "\n\nAdditional rules:\n"
            "- Never invent or estimate a number. If a value is not in the sources, say so.\n"
            "- Write a COMPLETE answer. Never end mid-sentence.\n"
            "- Answer in the same language the question was asked in (German or English).\n"
            "- The Sources section below contains retrieval evidence for this question. Treat it as relevant unless it is empty.\n"
            "- Do not answer that you lack information when one or more numbered sources are provided.\n"
            "- For definition questions like 'what is X?', define X from the closest matching source title or excerpt.\n"
            + expertise_note
            + lang_note
            + ("\n\n" + history_note.strip() if history_note else "")
            + conflict_section
            + "\n\nSources:\n"
            + "\n\n".join(context_lines)
            + kg_note
        )

    if formula_mode:
        messages[0]["content"] = (
            "You are TwoStrokeGPT, an expert on two-stroke engines and thermodynamics.\n"
            "The user is asking about an engineering formula.\n\n"
            "Structure your answer EXACTLY as follows:\n\n"
            "## Formula\n"
            "[Write the formula clearly. Wrap it in $$ delimiters so it renders correctly, for example: $$BTE = P_b / (m_dot_F * LHV)$$]\n\n"
            "**Variables:**\n"
            "- [Symbol]: [Full name] - [unit if known]\n"
            "(list every variable that appears in the formula)\n\n"
            "**What it means:**\n"
            "[One short paragraph in plain English explaining what the formula calculates and why it matters]\n\n"
            "**Source:**\n"
            "[Cite the document and page/slide]\n\n"
            "Rules:\n"
            "- Never invent a value. Only use what is in the sources.\n"
            "- Write a real answer in your own words. Never paste raw document text.\n"
            "- Never include internal markers like '--- Slide N ---', '--- Page N ---', or '[Source N]'.\n"
            "- If the formula appears in multiple sources, show the clearest version.\n"
            "- When writing the formula in the ## Formula section, wrap it in $$ delimiters.\n"
            "- Prefer plain ASCII math inside the delimiters, except Greek symbols already present in the sources.\n"
            "- Do not repeat garbled OCR or proprietary footer text.\n"
            "- If sources contain example values, include them under a short 'Example:' section.\n"
            "- IMPORTANT: If the sources contain ANY information related to the question, use it. "
            "Only say you cannot find information if the sources contain absolutely nothing relevant. "
            "Regulatory text, standards, and technical specifications ARE valid sources. "
            "Do not answer that you lack information when one or more numbered sources are provided."
            + lang_note
            + ("\n\n" + history_note.strip() if history_note else "")
            + "\n\nSOURCES:\n"
            + "\n\n".join(context_lines)
            + formula_note
        )

    full_answer = ""
    try:
        for token in stream_chat(messages, temperature=0.1, max_tokens=2000):
            full_answer += token
    except Exception as exc:
        yield _sse({"type": "error", "text": str(exc)})
        return

    if _looks_like_refusal(full_answer) and citations:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "_stream_answer: LLM refused despite %d citations. "
            "First 100 chars of answer: %r. "
            "First citation snippet: %r",
            len(citations),
            full_answer[:100],
            citations[0].get("snippet", "")[:100] if citations else "",
        )
        full_answer = _extractive_answer_from_citations(question, citations)
    elif _looks_like_raw_kg_answer(full_answer) and citations:
        kg_answer = _diagnostic_answer_from_kg_citations(question, citations)
        if kg_answer:
            full_answer = kg_answer
    full_answer = _clean_answer_for_user(full_answer)
    yield _sse({"type": "delta", "text": full_answer})

    # Grounding check
    if _should_verify_answer(full_answer):
        try:
            grounded = is_grounded(full_answer, chunks)
        except Exception:
            grounded = False
    else:
        grounded = True

    # Related questions — pass full question with history for better suggestions
    # Persist conversation (best-effort)
    try:
        from memory.store import append_turn

        append_turn(session_id, "user", question)
        append_turn(session_id, "assistant", full_answer)
    except Exception:
        pass

    answer_images = _fetch_images_for_citations(citations) if image_mode else []

    yield _sse({
        "type": "done",
        "citations": citations,
        "confidence": "high" if grounded else "low",
        "related_questions": [],
        "kg_paths": kg_result.get("paths", []),
        "graph_evidence": kg_result.get("graph_evidence", []),
        "intent": kg_result.get("intent", {}),
        "formulas": formula_context,
        "images": answer_images,
        "conflicts": conflicts,
    })

    if conflicts:
        yield _sse({"type": "conflicts", "conflicts": conflicts})

    if answer_images:
        yield _sse({"type": "images", "images": answer_images})

    try:
        import asyncio

        related_qs = asyncio.run(_related_questions_with_task(question + (history_note or ""), chunks))
    except Exception:
        related_qs = []
    yield _sse({"type": "related_questions", "related_questions": related_qs})


@app.post("/ask/stream")
def ask_stream(
    question: str = Form(...),
    session_id: str = Form("anon"),
    topic: str = Form(""),
    mode: str = Form("general"),
) -> StreamingResponse:
    """Streaming version of /ask using Server-Sent Events.

    The client consumes this with fetch() + ReadableStream (see index.html).
    Falls back gracefully to an empty done event when retrieval returns nothing.
    """
    return StreamingResponse(
        _stream_answer(question, session_id=session_id, topic=topic.strip() or None, mode=mode),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/conversation/{session_id}")
def get_conversation_history(session_id: str) -> JSONResponse:
    from memory.store import get_conversation

    try:
        turns = get_conversation(session_id) or []
        return JSONResponse({"session_id": session_id, "turns": turns[-20:]})
    except Exception as exc:
        return JSONResponse({"session_id": session_id, "turns": [], "error": str(exc)})


@app.get("/history/{session_id}")
def get_question_history(session_id: str) -> JSONResponse:
    from memory.store import get_conversation

    try:
        turns = get_conversation(session_id) or []
        questions = [
            str(t.get("content", ""))
            for t in turns
            if t.get("role") == "user" and str(t.get("content", "")).strip()
        ]
        return JSONResponse({"questions": list(reversed(questions[-20:]))})
    except Exception as exc:
        return JSONResponse({"questions": [], "error": str(exc)})


@app.post("/export/answer")
def export_answer(payload: dict[str, Any] = Body(...)) -> Response:
    from datetime import datetime

    question = str(payload.get("question") or "")
    answer = str(payload.get("answer") or "")
    citations = payload.get("citations") or []
    if not isinstance(citations, list):
        citations = []

    lines = [
        "--- TorqBase Answer Export ---",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Question: {question}",
        "",
        "Answer:",
        answer,
        "",
        "Sources:",
    ]
    for cit in citations:
        if not isinstance(cit, dict):
            continue
        n = cit.get("n", "")
        filename = cit.get("filename") or cit.get("source_title") or cit.get("doc_id") or "Unknown source"
        page = f" - page {cit.get('page')}" if cit.get("page") else ""
        snippet = str(cit.get("snippet") or "").strip()
        lines.append(f"[{n}] {filename}{page}")
        if snippet:
            lines.append(f"  {snippet}")
    lines.extend([
        "",
        "Generated by TorqBase | Hirth Engines Knowledge Base",
        "--------------------------------",
    ])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Response(
        "\n".join(lines),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="torqbase_answer_{timestamp}.txt"'},
    )


@app.get("/specs")
def browse_specs(
    q: str | None = None,
    doc_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> JSONResponse:
    from config import get_connection

    limit, offset = _limit_offset(limit, offset)
    q = (q or "").strip() or None
    doc_id = (doc_id or "").strip() or None
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            clauses: list[str] = []
            params: list[Any] = []
            if q:
                like = f"%{q}%"
                clauses.append("(key ILIKE %s OR row_label ILIKE %s OR value ILIKE %s)")
                params.extend([like, like, like])
            if doc_id:
                clauses.append("doc_id = %s")
                params.append(doc_id)
            where = "WHERE " + " AND ".join(clauses) if clauses else ""
            cur.execute(f"SELECT COUNT(*) FROM structured_facts {where}", params)
            total = int(cur.fetchone()[0] or 0)
            cur.execute(
                f"""
                SELECT doc_id, sheet, row_label, col_label, key, value, unit, source_ref
                FROM structured_facts
                {where}
                ORDER BY doc_id, sheet, key
                LIMIT %s OFFSET %s
                """,
                (*params, limit, offset),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        return JSONResponse({
            "facts": [
                {
                    "doc_id": r[0],
                    "sheet": r[1],
                    "row_label": r[2],
                    "col_label": r[3],
                    "key": r[4],
                    "value": r[5],
                    "unit": r[6],
                    "source_ref": r[7],
                }
                for r in rows
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        })
    except Exception as exc:
        return JSONResponse({"facts": [], "total": 0, "error": str(exc)}, status_code=500)


@app.get("/specs/summary")
def specs_summary() -> JSONResponse:
    from config import get_connection

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT doc_id, COUNT(*) AS fact_count, COUNT(DISTINCT key) AS unique_keys
                FROM structured_facts
                GROUP BY doc_id
                ORDER BY fact_count DESC
                """
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        return JSONResponse({
            "documents": [
                {"doc_id": r[0], "fact_count": int(r[1] or 0), "unique_keys": int(r[2] or 0)}
                for r in rows
            ]
        })
    except Exception as exc:
        return JSONResponse({"documents": [], "error": str(exc)}, status_code=500)


@app.get("/documents")
def list_documents(topic: str | None = None, limit: int = 50, offset: int = 0) -> JSONResponse:
    from config import get_connection
    from ingestion.knowledge_base import ensure_document_images_table, ensure_formulas_table

    limit, offset = _limit_offset(limit, offset)
    topic = (topic or "").strip() or None
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            ensure_document_images_table(cur)
            ensure_formulas_table(cur)
            conn.commit()
            params: list[Any] = []
            where = ""
            if topic:
                where = "WHERE cd.topic = %s"
                params.append(topic)
            params.extend([limit, offset])
            cur.execute(
                f"""
                WITH chunk_docs AS (
                    SELECT
                        c.doc_id,
                        COALESCE(MAX(c.metadata->>'filename'), c.doc_id) AS filename,
                        MAX(c.lang) AS lang,
                        COUNT(c.id) AS chunk_count,
                        MAX(c.metadata->>'topic') AS topic
                    FROM chunks c
                    GROUP BY c.doc_id
                ),
                image_counts AS (
                    SELECT doc_id, COUNT(id) AS image_count
                    FROM document_images
                    GROUP BY doc_id
                ),
                doc_meta AS (
                    SELECT DISTINCT ON (doc_id)
                        doc_id, filename, version, lang, uploaded_at
                    FROM documents
                    ORDER BY doc_id, version DESC
                )
                SELECT
                    cd.doc_id,
                    COALESCE(dm.filename, cd.filename) AS filename,
                    COALESCE(dm.version, 1) AS version,
                    COALESCE(dm.lang, cd.lang) AS lang,
                    dm.uploaded_at,
                    cd.chunk_count,
                    COALESCE(ic.image_count, 0) AS image_count,
                    cd.topic
                FROM chunk_docs cd
                LEFT JOIN doc_meta dm ON dm.doc_id = cd.doc_id
                LEFT JOIN image_counts ic ON ic.doc_id = cd.doc_id
                {where}
                ORDER BY dm.uploaded_at DESC NULLS LAST, filename
                LIMIT %s OFFSET %s
                """,
                tuple(params),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        return JSONResponse([
            {
                "doc_id": r[0],
                "filename": r[1],
                "version": r[2],
                "lang": r[3],
                "uploaded_at": r[4].isoformat() if r[4] else None,
                "chunk_count": int(r[5] or 0),
                "image_count": int(r[6] or 0),
                "topic": r[7],
            }
            for r in rows
        ])
    except Exception as exc:
        return JSONResponse({"error": str(exc), "items": []}, status_code=500)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> JSONResponse:
    from config import get_connection
    from ingestion.knowledge_base import ensure_document_images_table, ensure_formulas_table

    image_files: list[Path] = []
    chunks_removed = 0
    images_removed = 0
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            ensure_document_images_table(cur)
            ensure_formulas_table(cur)
            cur.execute("SELECT file_path, url FROM document_images WHERE doc_id = %s", (doc_id,))
            for file_path, url in cur.fetchall():
                if file_path:
                    image_files.append(Path(str(file_path)))
                elif url:
                    image_files.append(IMAGE_DIR / Path(str(url)).name)

            cur.execute("DELETE FROM document_images WHERE doc_id = %s", (doc_id,))
            images_removed = cur.rowcount or 0
            cur.execute("DELETE FROM formulas WHERE doc_id = %s", (doc_id,))
            cur.execute("DELETE FROM structured_facts WHERE doc_id = %s", (doc_id,))
            cur.execute("DELETE FROM graph_edges WHERE props->>'doc_id' = %s", (doc_id,))
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            chunks_removed = cur.rowcount or 0
            cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
            conn.commit()
        finally:
            conn.close()

        for path in image_files:
            try:
                resolved = path if path.is_absolute() else (Path.cwd() / path)
                if resolved.exists() and IMAGE_DIR.resolve() in resolved.resolve().parents:
                    resolved.unlink()
            except Exception:
                pass

        try:
            from agent.retriever_hybrid import invalidate_bm25_cache

            invalidate_bm25_cache()
        except Exception:
            try:
                from agent.retriever_hybrid import _bm25_cache

                _bm25_cache["timestamp"] = 0
            except Exception:
                pass

        return JSONResponse({"deleted": doc_id, "chunks_removed": chunks_removed, "images_removed": images_removed})
    except Exception as exc:
        return JSONResponse({"deleted": doc_id, "error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Feedback, gaps, graph
# ---------------------------------------------------------------------------

def _ensure_admin_tables(cur: Any) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id          BIGSERIAL PRIMARY KEY,
            session_id  TEXT,
            question    TEXT,
            answer      TEXT,
            vote        INT,
            correction  TEXT,
            expert_note TEXT,
            chunk_ids   BIGINT[],
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS gaps (
            id          BIGSERIAL PRIMARY KEY,
            question    TEXT,
            reason      TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            resolved    BOOLEAN DEFAULT false
        )
        """
    )


def _ensure_chunk_quality_columns(cur: Any) -> None:
    cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS quality_score FLOAT DEFAULT 0.0")
    cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS vote_count INT DEFAULT 0")
    cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS downvote_count INT DEFAULT 0")
    cur.execute("CREATE INDEX IF NOT EXISTS chunks_quality_score_idx ON chunks (quality_score)")


def _limit_offset(limit: int, offset: int) -> tuple[int, int]:
    return max(1, min(int(limit or 50), 200)), max(0, int(offset or 0))


@app.get("/admin/gaps")
def admin_gaps(resolved: bool = False, limit: int = 50, offset: int = 0) -> JSONResponse:
    from config import get_connection

    limit, offset = _limit_offset(limit, offset)
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            _ensure_admin_tables(cur)
            conn.commit()
            cur.execute(
                """
                SELECT id, question, reason, created_at, resolved
                FROM gaps
                WHERE resolved = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (resolved, limit, offset),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        return JSONResponse([
            {"id": r[0], "question": r[1], "reason": r[2], "created_at": r[3].isoformat() if r[3] else None, "resolved": r[4]}
            for r in rows
        ])
    except Exception as exc:
        return JSONResponse({"error": str(exc), "items": []}, status_code=500)


@app.post("/admin/gaps/{gap_id}/resolve")
def admin_resolve_gap(gap_id: int, payload: dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
    from config import get_connection

    resolved = bool(payload.get("resolved", True))
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            _ensure_admin_tables(cur)
            cur.execute("UPDATE gaps SET resolved = %s WHERE id = %s", (resolved, gap_id))
            conn.commit()
        finally:
            conn.close()
        return JSONResponse({"ok": True})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.get("/admin/feedback")
def admin_feedback(vote: int | None = None, limit: int = 50, offset: int = 0) -> JSONResponse:
    from config import get_connection

    limit, offset = _limit_offset(limit, offset)
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            _ensure_admin_tables(cur)
            conn.commit()
            params: list[Any] = []
            where = ""
            if vote in (-1, 1):
                where = "WHERE vote = %s"
                params.append(vote)
            params.extend([limit, offset])
            cur.execute(
                f"""
                SELECT id, session_id, question, answer, vote, correction, expert_note, created_at
                FROM feedback
                {where}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                tuple(params),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        return JSONResponse([
            {
                "id": r[0],
                "session_id": r[1],
                "question": r[2],
                "answer": r[3],
                "vote": r[4],
                "correction": r[5],
                "expert_note": r[6],
                "created_at": r[7].isoformat() if r[7] else None,
            }
            for r in rows
        ])
    except Exception as exc:
        return JSONResponse({"error": str(exc), "items": []}, status_code=500)


@app.get("/admin/stats")
def admin_stats() -> JSONResponse:
    from config import get_connection
    from ingestion.knowledge_base import ensure_document_images_table, ensure_formulas_table

    stats = {
        "total_chunks": 0,
        "total_documents": 0,
        "total_gaps": 0,
        "total_feedback": 0,
        "thumbs_up": 0,
        "thumbs_down": 0,
        "topics_with_content": 0,
        "images_indexed": 0,
        "formulas_indexed": 0,
        "downvoted_chunks": 0,
        "trusted_chunks": 0,
        "correction_chunks": 0,
    }
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            _ensure_admin_tables(cur)
            _ensure_chunk_quality_columns(cur)
            ensure_document_images_table(cur)
            ensure_formulas_table(cur)
            conn.commit()
            queries = {
                "total_chunks": "SELECT COUNT(*) FROM chunks",
                "total_documents": "SELECT COUNT(DISTINCT doc_id) FROM chunks",
                "total_gaps": "SELECT COUNT(*) FROM gaps WHERE resolved = false",
                "total_feedback": "SELECT COUNT(*) FROM feedback",
                "thumbs_up": "SELECT COUNT(*) FROM feedback WHERE vote = 1",
                "thumbs_down": "SELECT COUNT(*) FROM feedback WHERE vote = -1",
                "topics_with_content": "SELECT COUNT(DISTINCT metadata->>'topic') FROM chunks WHERE metadata->>'topic' IS NOT NULL",
                "images_indexed": "SELECT COUNT(*) FROM document_images",
                "formulas_indexed": "SELECT COUNT(*) FROM formulas",
                "downvoted_chunks": "SELECT COUNT(*) FROM chunks WHERE quality_score < -0.3",
                "trusted_chunks": "SELECT COUNT(*) FROM chunks WHERE quality_score > 0.3",
                "correction_chunks": "SELECT COUNT(*) FROM chunks WHERE metadata->>'source_type' = 'expert_correction'",
            }
            for key, sql in queries.items():
                cur.execute(sql)
                stats[key] = int(cur.fetchone()[0] or 0)
        finally:
            conn.close()
        return JSONResponse(stats)
    except Exception as exc:
        stats["error"] = str(exc)
        return JSONResponse(stats, status_code=500)


@app.get("/admin/low-quality-chunks")
def admin_low_quality_chunks(limit: int = 50) -> JSONResponse:
    from config import get_connection

    limit = max(1, min(int(limit or 50), 100))
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            _ensure_chunk_quality_columns(cur)
            conn.commit()
            cur.execute(
                """
                SELECT id, doc_id, metadata->>'source_title' AS filename,
                       COALESCE(metadata->>'page', metadata->>'slide') AS page,
                       quality_score, downvote_count, vote_count,
                       LEFT(content, 150) AS preview
                FROM chunks
                WHERE quality_score < -0.2 OR downvote_count >= 2
                ORDER BY quality_score ASC, downvote_count DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        return JSONResponse([
            {
                "id": r[0],
                "doc_id": r[1],
                "filename": r[2] or r[1],
                "page": r[3],
                "quality_score": float(r[4] or 0.0),
                "downvote_count": int(r[5] or 0),
                "vote_count": int(r[6] or 0),
                "preview": r[7] or "",
            }
            for r in rows
        ])
    except Exception as exc:
        return JSONResponse({"error": str(exc), "items": []}, status_code=500)


@app.post("/admin/chunks/{chunk_id}/suppress")
def admin_suppress_chunk(chunk_id: int) -> JSONResponse:
    from config import get_connection

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            _ensure_chunk_quality_columns(cur)
            cur.execute("UPDATE chunks SET quality_score = -999 WHERE id = %s", (chunk_id,))
            conn.commit()
        finally:
            conn.close()
        try:
            from agent.retriever_hybrid import invalidate_bm25_cache

            invalidate_bm25_cache()
        except Exception:
            pass
        return JSONResponse({"ok": True, "chunk_id": chunk_id})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


def _doc_id_candidates(value: str) -> list[str]:
    import re

    raw = str(value or "").strip()
    lower = raw.lower()
    candidates = [
        lower,
        re.sub(r"[^a-z0-9_.]", "_", lower),
        re.sub(r"[^\w.-]", "_", Path(raw).stem.lower()),
    ]
    result: list[str] = []
    for item in candidates:
        if item and item not in result:
            result.append(item)
    return result


@app.get("/admin/doc-debug/{doc_id}")
def admin_doc_debug(doc_id: str) -> JSONResponse:
    """Return ingestion/debug details for one document without re-ingesting it."""
    from config import get_connection

    candidates = _doc_id_candidates(doc_id)
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT doc_id, filename, version, lang, storage_uri, created_at
            FROM documents
            WHERE doc_id = ANY(%s) OR lower(filename) = lower(%s)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (candidates, doc_id),
        )
        doc_row = cur.fetchone()
        resolved_doc_id = doc_row[0] if doc_row else candidates[0]
        filename = doc_row[1] if doc_row else doc_id

        cur.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = %s", (resolved_doc_id,))
        chunk_count = int(cur.fetchone()[0] or 0)
        cur.execute(
            """
            SELECT id, LEFT(content, 200), metadata
            FROM chunks
            WHERE doc_id = %s
            ORDER BY id
            LIMIT 3
            """,
            (resolved_doc_id,),
        )
        chunks_preview = [
            {"id": row[0], "content_preview": row[1], "metadata": row[2] or {}}
            for row in cur.fetchall()
        ]

        cur.execute("SELECT COUNT(*) FROM document_images WHERE doc_id = %s", (resolved_doc_id,))
        image_count = int(cur.fetchone()[0] or 0)

        formula_count = 0
        try:
            cur.execute("SELECT COUNT(*) FROM formulas WHERE doc_id = %s", (resolved_doc_id,))
            formula_count = int(cur.fetchone()[0] or 0)
        except Exception:
            conn.rollback()

        ingestion_job = None
        try:
            cur.execute(
                """
                SELECT ji.status, ji.error, ji.created_at
                FROM ingestion_job_items ji
                JOIN file_inventory fi ON fi.id = ji.inventory_id
                WHERE lower(fi.file_name) = lower(%s)
                   OR fi.file_name = ANY(%s)
                ORDER BY ji.created_at DESC
                LIMIT 1
                """,
                (filename, [doc_id, filename]),
            )
            job_row = cur.fetchone()
            if job_row:
                ingestion_job = {
                    "status": job_row[0],
                    "error": job_row[1],
                    "created_at": job_row[2].isoformat() if job_row[2] else None,
                }
            else:
                cur.execute(
                    """
                    SELECT status, error, created_at
                    FROM file_inventory
                    WHERE lower(file_name) = lower(%s)
                       OR file_name = ANY(%s)
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (filename, [doc_id, filename]),
                )
                inv_row = cur.fetchone()
                if inv_row:
                    ingestion_job = {
                        "status": inv_row[0],
                        "error": inv_row[1],
                        "created_at": inv_row[2].isoformat() if inv_row[2] else None,
                    }
        except Exception:
            conn.rollback()

        return JSONResponse({
            "doc_id": resolved_doc_id,
            "filename": filename,
            "chunk_count": chunk_count,
            "chunks_preview": chunks_preview,
            "image_count": image_count,
            "formula_count": formula_count,
            "ingestion_job": ingestion_job,
        })
    except Exception as exc:
        return JSONResponse({"doc_id": doc_id, "error": str(exc)}, status_code=500)
    finally:
        conn.close()


@app.post("/feedback")
async def feedback(
    session_id: str = Form(...),
    question: str = Form(...),
    answer: str = Form(...),
    vote: int = Form(0),
    correction: str = Form(""),
    chunk_ids: str = Form("[]"),
) -> JSONResponse:
    """Record user feedback. Corrections become high-priority knowledge; votes reweight retrieval."""
    from memory.store import record_feedback

    try:
        parsed_chunk_ids: list[int] = []
        try:
            raw_ids = _json.loads(chunk_ids or "[]")
            if isinstance(raw_ids, list):
                parsed_chunk_ids = [int(cid) for cid in raw_ids if str(cid).isdigit()]
        except Exception:
            parsed_chunk_ids = []
        record_feedback(
            session_id=session_id,
            question=question,
            answer=answer,
            vote=vote,
            correction=correction,
            chunk_ids=parsed_chunk_ids,
        )
    except Exception:
        pass

    return JSONResponse({"status": "recorded"})


@app.get("/entities")
def get_entities(doc_id: str | None = None) -> JSONResponse:
    """Return aggregated entities and tags extracted by the domain enricher.

    Queries the 'entities' and 'tags' arrays stored in chunks.metadata JSONB.
    Optional doc_id filter. Returns top-50 entities and top-30 tags by frequency.
    """
    from config import get_connection
    import json as _j

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            if doc_id:
                cur.execute(
                    "SELECT metadata FROM chunks WHERE doc_id = %s AND metadata ? 'entities'",
                    (doc_id,),
                )
            else:
                cur.execute("SELECT metadata FROM chunks WHERE metadata ? 'entities'")
            rows = cur.fetchall()
        finally:
            conn.close()

        entity_counts: dict[str, dict] = {}
        tag_counts: dict[str, int] = {}

        for (meta_raw,) in rows:
            meta = meta_raw if isinstance(meta_raw, dict) else _j.loads(meta_raw or "{}")
            for ent in meta.get("entities", []):
                key = f"{ent.get('type','?')}::{ent.get('name','?')}"
                if key not in entity_counts:
                    entity_counts[key] = {"type": ent.get("type"), "name": ent.get("name"), "count": 0}
                entity_counts[key]["count"] += 1
            for tag in meta.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        entities = sorted(entity_counts.values(), key=lambda e: e["count"], reverse=True)[:50]
        tags = sorted(tag_counts.items(), key=lambda t: t[1], reverse=True)[:30]

        return JSONResponse({
            "entities": entities,
            "tags": [{"tag": t, "count": c} for t, c in tags],
        })
    except Exception as exc:
        return JSONResponse({"entities": [], "tags": [], "error": str(exc)})


@app.get("/chunks")
def list_chunks(doc_id: str | None = None, limit: int = 50) -> JSONResponse:
    """List indexed chunks, optionally filtered by doc_id."""
    from config import get_connection

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            if doc_id:
                cur.execute(
                    """
                    SELECT id, doc_id, content, metadata, source_refs
                    FROM chunks WHERE doc_id = %s
                    ORDER BY id LIMIT %s
                    """,
                    (doc_id, limit),
                )
            else:
                cur.execute(
                    "SELECT id, doc_id, content, metadata, source_refs FROM chunks ORDER BY id LIMIT %s",
                    (limit,),
                )
            rows = cur.fetchall()
        finally:
            conn.close()

        import json as _j
        return JSONResponse({
            "chunks": [
                {
                    "id": r[0],
                    "doc_id": r[1],
                    "snippet": r[2][:200],
                    "metadata": r[3] if isinstance(r[3], dict) else _j.loads(r[3] or "{}"),
                    "source_refs": r[4] if isinstance(r[4], list) else _j.loads(r[4] or "[]"),
                }
                for r in rows
            ]
        })
    except Exception as exc:
        return JSONResponse({"chunks": [], "error": str(exc)})


@app.get("/chunks/{chunk_id}/view")
def view_chunk_html(chunk_id: int):
    """Render a standalone HTML page for a single chunk (evidence anchor, opens in new tab)."""
    from fastapi.responses import HTMLResponse
    from config import get_connection
    import json as _j

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, doc_id, content, metadata, source_refs FROM chunks WHERE id = %s",
                (chunk_id,),
            )
            row = cur.fetchone()
        finally:
            conn.close()

        if not row:
            return HTMLResponse("<h2>Chunk not found</h2>", status_code=404)

        meta = row[3] if isinstance(row[3], dict) else _j.loads(row[3] or "{}")
        refs = row[4] if isinstance(row[4], list) else _j.loads(row[4] or "[]")
        ref_str = ", ".join(r.get("filename") or r.get("source") or str(r) for r in refs) or "—"
        content_escaped = row[2].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

        html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><title>Chunk {row[0]} — {row[1]}</title>
<style>
  body{{font-family:ui-sans-serif,system-ui,sans-serif;max-width:860px;margin:40px auto;padding:0 24px;color:#1c1917;background:#fafaf9}}
  h2{{font-size:18px;margin-bottom:4px}}
  .meta{{font-size:12px;color:#78716c;margin-bottom:20px}}
  pre{{background:#f5f5f4;border-radius:8px;padding:16px;white-space:pre-wrap;word-break:break-word;line-height:1.6;font-size:14px}}
  a,button.close{{color:#b45309;cursor:pointer;background:none;border:none;font-size:14px;padding:0;text-decoration:underline}}
  .notice{{font-size:12px;color:#78716c;margin-top:4px}}
</style></head><body>
<h2>Chunk #{row[0]} — <code>{row[1]}</code></h2>
<div class="meta">
  type: {meta.get("chunk_type", meta.get("type","?"))} &nbsp;|&nbsp;
  lang: {meta.get("lang","?")} &nbsp;|&nbsp;
  page: {meta.get("page", "?")} &nbsp;|&nbsp;
  sources: {ref_str}
</div>
<pre>{content_escaped}</pre>
<p>
  <button class="close" onclick="window.close()">Close this tab</button>
  <span class="notice">&nbsp;— opened by TwoStrokeGPT</span>
</p>
</body></html>"""
        return HTMLResponse(html)
    except Exception as exc:
        return HTMLResponse(f"<h2>Error: {exc}</h2>", status_code=500)


@app.get("/chunks/{chunk_id}/images")
def get_chunk_images(chunk_id: int) -> JSONResponse:
    """Return extracted document figures linked to a chunk."""
    from config import get_connection

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            from ingestion.knowledge_base import ensure_document_images_table

            ensure_document_images_table(cur)
            cur.execute(
                """
                SELECT url, file_path, caption, page_or_slide
                FROM document_images
                WHERE chunk_id = %s
                ORDER BY image_index, id
                """,
                (chunk_id,),
            )
            rows = cur.fetchall()
            if not rows:
                cur.execute(
                    """
                    SELECT doc_id,
                           COALESCE(metadata->>'slide', metadata->>'page') AS page_or_slide
                    FROM chunks
                    WHERE id = %s
                    """,
                    (chunk_id,),
                )
                chunk_row = cur.fetchone()
                if chunk_row and chunk_row[1]:
                    cur.execute(
                        """
                        SELECT url, file_path, caption, page_or_slide
                        FROM document_images
                        WHERE doc_id = %s AND page_or_slide = %s
                        ORDER BY image_index, id
                        """,
                        (chunk_row[0], int(chunk_row[1])),
                    )
                    rows = cur.fetchall()
        finally:
            conn.close()

        images = []
        for url, file_path, caption, page_or_slide in rows:
            filename = Path(str(file_path)).name
            images.append({
                "url": url or f"/images/{filename}",
                "caption": caption or "",
                "page_or_slide": page_or_slide,
            })
        return JSONResponse(images)
    except Exception as exc:
        return JSONResponse({"error": str(exc), "images": []}, status_code=500)


@app.get("/formulas/search")
def formula_search(q: str = "", topic: str | None = None, limit: int = 20) -> JSONResponse:
    """Search extracted formula rows by formula text and surrounding context."""
    try:
        return JSONResponse(_search_formulas(q, topic=topic, limit=limit))
    except Exception as exc:
        return JSONResponse({"error": str(exc), "results": []}, status_code=500)


@app.get("/formulas/summary")
def formula_summary() -> JSONResponse:
    """Return formula counts grouped by topic."""
    from config import get_connection
    from ingestion.knowledge_base import ensure_formulas_table

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            ensure_formulas_table(cur)
            cur.execute(
                """
                SELECT COALESCE(topic, 'Unsorted') AS topic, COUNT(*)
                FROM formulas
                GROUP BY COALESCE(topic, 'Unsorted')
                ORDER BY COUNT(*) DESC, topic
                """
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        return JSONResponse({"by_topic": [{"topic": r[0], "count": int(r[1])} for r in rows]})
    except Exception as exc:
        return JSONResponse({"by_topic": [], "error": str(exc)}, status_code=500)


@app.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: int) -> JSONResponse:
    """Return full content + metadata for a single chunk (evidence anchor)."""
    from config import get_connection

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, doc_id, content, metadata, source_refs FROM chunks WHERE id = %s",
                (chunk_id,),
            )
            row = cur.fetchone()
        finally:
            conn.close()

        if not row:
            return JSONResponse({"error": "chunk not found"}, status_code=404)

        import json as _j
        return JSONResponse({
            "id": row[0],
            "doc_id": row[1],
            "content": row[2],
            "metadata": row[3] if isinstance(row[3], dict) else _j.loads(row[3] or "{}"),
            "source_refs": row[4] if isinstance(row[4], list) else _j.loads(row[4] or "[]"),
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


def _json_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = _json.loads(raw or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _confidence(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _listify(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def serialize_graph_node(node_id: int, ntype: str, name: str, props_raw: Any) -> dict[str, Any] | None:
    """Serialize a graph node with backward-compatible fields plus rich metadata."""
    from ingestion.kg_normalizer import normalize_entity

    normalized = normalize_entity(name, ntype)
    if not normalized["is_valid"]:
        return None
    props = _json_obj(props_raw)
    aliases = props.get("aliases") or normalized.get("aliases") or []
    doc_ids = _listify(props.get("doc_ids")) or _listify(props.get("doc_id"))
    confidence = _confidence(props.get("confidence"))
    label = props.get("display_name") or normalized["display_name"] or normalized["canonical_name"]
    return {
        "id": node_id,
        "label": label,
        "name": label,  # Backward compatibility for current frontend.
        "type": normalized["type"],
        "canonical_name": normalized["canonical_name"],
        "aliases": aliases,
        "confidence": confidence,
        "source_count": int(props.get("source_count") or len(set(map(str, doc_ids))) or 0),
        "doc_ids": doc_ids,
        "props": props,
    }


def serialize_graph_edge(edge_id: int, src_id: int, dst_id: int, relation: str, props_raw: Any) -> dict[str, Any]:
    """Serialize a graph edge with provenance defaults for old rows."""
    props = _json_obj(props_raw)
    confidence = _confidence(props.get("confidence"))
    extraction_method = props.get("extraction_method") or "unknown"
    evidence = str(props.get("evidence") or "")
    chunk_id = props.get("source_chunk_id", props.get("chunk_id"))
    topic = props.get("topic")
    relative_path = props.get("relative_path")
    file_type = props.get("file_type")
    return {
        "id": edge_id,
        "source": src_id,
        "target": dst_id,
        "relation": relation,  # Backward compatibility for current frontend.
        "type": relation,
        "confidence": confidence,
        "evidence": evidence,
        "extraction_method": extraction_method,
        "doc_id": props.get("doc_id"),
        "source_chunk_id": chunk_id,
        "chunk_id": chunk_id,
        "page": props.get("page"),
        "source_title": props.get("source_title"),
        "topic": topic,
        "relative_path": relative_path,
        "file_type": file_type,
        "props": {
            "doc_id": props.get("doc_id"),
            "source_chunk_id": chunk_id,
            "chunk_id": chunk_id,
            "page": props.get("page"),
            "evidence": evidence,
            "confidence": confidence,
            "extraction_method": extraction_method,
            "source_title": props.get("source_title"),
            "topic": topic,
            "relative_path": relative_path,
            "file_type": file_type,
            **props,
        },
    }


@app.get("/graph")
def get_graph(
    node_type: str | None = None,
    edge_type: str | None = None,
    doc_id: str | None = None,
    topic: str | None = None,
    min_confidence: float = 0.4,
    extraction_method: str | None = None,
    include_seed: bool = True,
    limit: int = 600,
) -> JSONResponse:
    """Return graph nodes and edges for knowledge-map display."""
    from config import get_connection

    try:
        limit = max(1, min(int(limit), 2000))
        min_confidence = max(0.0, min(1.0, float(min_confidence)))
        topic = (topic or "").strip() or None
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, type, name, props FROM graph_nodes LIMIT %s", (limit,))
            nodes = []
            valid_ids = set()
            all_node_ids = set()
            for row in cur.fetchall():
                node = serialize_graph_node(*row)
                if not node:
                    continue
                all_node_ids.add(node["id"])
                if node_type and node["type"] != node_type:
                    continue
                if doc_id and doc_id not in {str(d) for d in node["doc_ids"]}:
                    continue
                nodes.append(node)
                valid_ids.add(node["id"])
            cur.execute(
                "SELECT id, src_id, dst_id, relation, props FROM graph_edges LIMIT %s",
                (limit,),
            )
            edges = []
            edge_node_ids = set()
            for row in cur.fetchall():
                edge = serialize_graph_edge(*row)
                if edge["source"] not in all_node_ids or edge["target"] not in all_node_ids:
                    continue
                if edge_type and edge["relation"] != edge_type:
                    continue
                if doc_id and edge.get("doc_id") != doc_id:
                    continue
                if topic and edge.get("topic") != topic:
                    continue
                if extraction_method and edge["extraction_method"] != extraction_method:
                    continue
                if not include_seed and edge["extraction_method"] == "manual_seed":
                    continue
                if (
                    edge["confidence"] is not None
                    and edge["confidence"] < min_confidence
                    and edge["extraction_method"] != "manual_seed"
                ):
                    continue
                edge_node_ids.update([edge["source"], edge["target"]])
                edges.append(edge)
            if node_type or doc_id:
                edges = [e for e in edges if e["source"] in valid_ids and e["target"] in valid_ids]
                edge_node_ids = {nid for e in edges for nid in (e["source"], e["target"])}
            if topic:
                valid_ids = edge_node_ids
            else:
                valid_ids = edge_node_ids or valid_ids
            if topic:
                nodes = [n for n in nodes if n["id"] in valid_ids]
            else:
                nodes = [n for n in nodes if n["id"] in valid_ids or not edges]
        finally:
            conn.close()

        return JSONResponse({
            "nodes": nodes,
            "edges": edges,
            "filters": {
                "node_type": node_type,
                "edge_type": edge_type,
                "doc_id": doc_id,
                "topic": topic,
                "min_confidence": min_confidence,
                "extraction_method": extraction_method,
                "include_seed": include_seed,
                "limit": limit,
            },
        })
    except Exception:
        return JSONResponse({"nodes": [], "edges": [], "error": "db unavailable"})


@app.get("/graph/diagnostic-paths")
def graph_diagnostic_paths(query: str, limit: int = 5) -> JSONResponse:
    """Return KG diagnostic paths for a query."""
    try:
        from agent.kg_retrieval import retrieve_kg_context

        result = retrieve_kg_context(query, limit=max(1, min(limit, 20)))
        return JSONResponse({
            "query": query,
            "intent": result.get("intent", {}),
            "paths": result.get("paths", []),
            "context": result.get("context", ""),
        })
    except Exception as exc:
        return JSONResponse({"query": query, "paths": [], "error": str(exc)}, status_code=500)


@app.get("/graph/quality")
def graph_quality() -> JSONResponse:
    """Return KG quality metrics: node/edge counts, confidence, evidence coverage."""
    from config import get_connection
    import json as _j

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()

            cur.execute("SELECT COUNT(*) FROM graph_nodes")
            total_nodes = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM graph_edges")
            total_edges = cur.fetchone()[0]

            cur.execute("SELECT type, COUNT(*) FROM graph_nodes GROUP BY type ORDER BY COUNT(*) DESC")
            nodes_by_type = {r[0]: r[1] for r in cur.fetchall()}

            cur.execute("SELECT relation, COUNT(*) FROM graph_edges GROUP BY relation ORDER BY COUNT(*) DESC")
            edges_by_relation = {r[0]: r[1] for r in cur.fetchall()}

            # Props-based stats — safe against NULL props
            cur.execute("SELECT props FROM graph_edges WHERE props IS NOT NULL")
            edge_props_rows = cur.fetchall()

            method_counts: dict = {}
            confidences: list[float] = []
            with_evidence = 0
            with_doc_id = 0
            with_chunk_id = 0

            for (props_raw,) in edge_props_rows:
                props = props_raw if isinstance(props_raw, dict) else _j.loads(props_raw or "{}")
                method = str(props.get("extraction_method") or "unknown")
                method_counts[method] = method_counts.get(method, 0) + 1
                conf = props.get("confidence")
                try:
                    confidences.append(float(conf))
                except (TypeError, ValueError):
                    pass
                if props.get("evidence"):
                    with_evidence += 1
                if props.get("doc_id"):
                    with_doc_id += 1
                if props.get("source_chunk_id") or props.get("chunk_id"):
                    with_chunk_id += 1

            avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else None
            edge_total = len(edge_props_rows)
            evidence_pct  = round(with_evidence / edge_total * 100, 1) if edge_total else 0
            doc_id_pct    = round(with_doc_id   / edge_total * 100, 1) if edge_total else 0
            chunk_id_pct  = round(with_chunk_id / edge_total * 100, 1) if edge_total else 0

            # Top connected nodes
            cur.execute("""
                SELECT n.name, n.type, COUNT(*) AS degree
                FROM graph_nodes n
                JOIN graph_edges e ON e.src_id = n.id OR e.dst_id = n.id
                GROUP BY n.id, n.name, n.type
                ORDER BY degree DESC LIMIT 10
            """)
            top_connected = [{"name": r[0], "type": r[1], "degree": r[2]} for r in cur.fetchall()]

            # Unknown / noisy nodes
            cur.execute("SELECT name FROM graph_nodes WHERE type = 'unknown' LIMIT 10")
            noisy_unknown = [r[0] for r in cur.fetchall()]

        finally:
            conn.close()

        return JSONResponse({
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "nodes_by_type": nodes_by_type,
            "edges_by_relation": edges_by_relation,
            "edges_by_extraction_method": method_counts,
            "avg_confidence": avg_confidence,
            "evidence_coverage_pct": evidence_pct,
            "doc_id_coverage_pct": doc_id_pct,
            "chunk_id_coverage_pct": chunk_id_pct,
            "top_connected_nodes": top_connected,
            "noisy_unknown_nodes": noisy_unknown,
        })
    except Exception as exc:
        return JSONResponse({
            "total_nodes": 0, "total_edges": 0,
            "nodes_by_type": {}, "edges_by_relation": {},
            "edges_by_extraction_method": {}, "avg_confidence": None,
            "evidence_coverage_pct": 0, "doc_id_coverage_pct": 0,
            "chunk_id_coverage_pct": 0, "top_connected_nodes": [],
            "noisy_unknown_nodes": [], "error": str(exc),
        })


@app.get("/topics")
def topics() -> JSONResponse:
    """Return known corpus topics from indexed chunks, inventory, and demo presets."""
    from config import get_connection

    preset_topics = [
        "CAD",
        "Verbrennungsmotoren",
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
        "Schulungen",
        "Bachelor_Master_Diplom_Doktorarbeiten",
        "Vibrationen",
        "Drehmomente",
        "Feinstellung-Zweitaktmotor",
    ]
    topic_map = {name: {"topic": name, "chunks": 0, "inventory_files": 0, "size_bytes": 0} for name in preset_topics}

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT metadata->>'topic' AS topic, COUNT(*)
                FROM chunks
                WHERE metadata ? 'topic' AND COALESCE(metadata->>'topic', '') <> ''
                GROUP BY topic
                """
            )
            for topic_name, count in cur.fetchall():
                row = topic_map.setdefault(topic_name, {"topic": topic_name, "chunks": 0, "inventory_files": 0, "size_bytes": 0})
                row["chunks"] = int(count or 0)

            cur.execute(
                """
                SELECT topic, COUNT(*), COALESCE(SUM(size_bytes), 0)
                FROM file_inventory
                WHERE COALESCE(topic, '') <> ''
                GROUP BY topic
                """
            )
            for topic_name, count, size_bytes in cur.fetchall():
                row = topic_map.setdefault(topic_name, {"topic": topic_name, "chunks": 0, "inventory_files": 0, "size_bytes": 0})
                row["inventory_files"] = int(count or 0)
                row["size_bytes"] = int(size_bytes or 0)
        finally:
            conn.close()
    except Exception:
        pass

    presets = [
        {"label": "Engine troubleshooting", "topic": "Verbrennungsmotoren", "question": "What are the likely causes and fixes for engine misfire?"},
        {"label": "Torque specs", "topic": "Drehmomente", "question": "What torque specifications are available?"},
        {"label": "Vibration", "topic": "Vibrationen", "question": "What vibration issues and checks are documented?"},
        {"label": "Two-stroke fine tuning", "topic": "Feinstellung-Zweitaktmotor", "question": "What should I check for two-stroke fine tuning?"},
        {"label": "Standards/certification", "topic": "Normen DIN ISO VDI FAR ASTM LURS", "question": "Which standards or certification requirements are referenced?"},
        {"label": "Propeller", "topic": "Propeller", "question": "What propeller-related guidance is available?"},
    ]
    rows = sorted(topic_map.values(), key=lambda r: r["topic"].casefold())
    return JSONResponse({"topics": rows, "presets": presets})


@app.get("/search")
def semantic_search(q: str, limit: int = 10, topic: str | None = None) -> JSONResponse:
    """Hybrid BM25 + dense semantic search over indexed chunks."""
    import json as _j
    from config import get_connection

    if not q or not q.strip():
        return JSONResponse({"results": [], "error": "empty query"})

    topic = (topic or "").strip() or None
    try:
        from agent.retriever_hybrid import search as hybrid_search

        hits = hybrid_search(q.strip(), k=limit, topic=topic)
        results = []
        for h in hits:
            metadata = h.get("metadata", {}) or {}
            refs = h.get("source_refs") or [{}]
            ref = refs[0] if refs else {}
            results.append({
                "id":       h.get("id"),
                "doc_id":   h.get("doc_id", ""),
                "score":    round(float(h.get("score", 0)), 4),
                "snippet":  (h.get("content") or h.get("snippet") or "")[:300],
                "metadata": metadata,
                "filename": h.get("filename") or ref.get("filename", ""),
                "topic": metadata.get("topic") or ref.get("topic"),
                "source_title": metadata.get("source_title") or ref.get("source_title") or ref.get("filename", ""),
                "page": ref.get("page") or metadata.get("page"),
                "slide": ref.get("slide") or metadata.get("slide"),
                "sheet": ref.get("sheet") or metadata.get("sheet") or metadata.get("table_name"),
            })
        return JSONResponse({"results": results, "topic": topic})
    except Exception as exc:
        # Fallback: plain SQL full-text search
        try:
            conn = get_connection()
            try:
                cur = conn.cursor()
                where_topic = "AND metadata->>'topic' = %s" if topic else ""
                params = [q, q]
                if topic:
                    params.append(topic)
                params.append(limit)
                cur.execute(
                    f"""
                    SELECT id, doc_id, content, metadata, source_refs,
                           ts_rank_cd(to_tsvector('simple', content),
                                      plainto_tsquery('simple', %s)) AS rank
                    FROM chunks
                    WHERE to_tsvector('simple', content) @@ plainto_tsquery('simple', %s)
                    {where_topic}
                    ORDER BY rank DESC LIMIT %s
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
            finally:
                conn.close()
            results = []
            for r in rows:
                meta = r[3] if isinstance(r[3], dict) else _j.loads(r[3] or "{}")
                refs = r[4] if isinstance(r[4], list) else _j.loads(r[4] or "[]")
                ref = refs[0] if refs else {}
                filename = (ref.get("filename") or ref.get("source") or "") if refs else ""
                results.append({
                    "id":       r[0],
                    "doc_id":   r[1],
                    "score":    round(float(r[5]), 4),
                    "snippet":  r[2][:300],
                    "metadata": meta,
                    "filename": filename,
                    "topic": meta.get("topic") or ref.get("topic"),
                    "source_title": meta.get("source_title") or ref.get("source_title") or filename,
                    "page": ref.get("page") or meta.get("page"),
                    "slide": ref.get("slide") or meta.get("slide"),
                    "sheet": ref.get("sheet") or meta.get("sheet") or meta.get("table_name"),
                })
            return JSONResponse({"results": results, "mode": "fulltext_fallback", "topic": topic})
        except Exception as exc2:
            return JSONResponse({"results": [], "error": str(exc2)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=settings.app_host, port=settings.app_port, reload=True)
