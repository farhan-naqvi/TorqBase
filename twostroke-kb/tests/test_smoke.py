"""Smoke tests — confirm the app imports and basic routing works before features land."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def test_config_loads():
    from config import get_settings

    s = get_settings()
    assert s.embedding_dim > 0
    assert s.rerank_top_k <= s.retrieve_top_k


def test_format_router_known_ext():
    from ingestion.format_router import EXT_MAP

    assert EXT_MAP[".pdf"] == "pdf_or_ocr"
    assert EXT_MAP[".xlsx"] == "sheet"
    assert EXT_MAP[".url"] == "link"


def test_topic_inference_and_boost():
    from agent.retriever_hybrid import _apply_topic_boost, infer_topic

    assert infer_topic("What torque specs are available?") == "Drehmomente"
    assert infer_topic("Show propeller guidance") == "Propeller"
    chunks = [
        {"id": 1, "score": 0.5, "metadata": {"topic": "Drehmomente"}, "source_refs": []},
        {"id": 2, "score": 0.51, "metadata": {"topic": "Propeller"}, "source_refs": []},
    ]
    boosted = _apply_topic_boost(chunks, "Drehmomente")
    assert boosted[0]["id"] == 1
    assert boosted[0]["topic_boost"] == "Drehmomente"


def test_link_handler_extracts_url(tmp_path):
    from ingestion.parsers import link_handler

    f = tmp_path / "HYDAC International.url"
    f.write_text("[InternetShortcut]\nURL=https://www.hydac.com\n")
    doc = link_handler.parse(f)
    assert doc.metadata["url"] == "https://www.hydac.com"
    assert doc.metadata["fetched"] is False


def test_app_health():
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    assert client.get("/health").json() == {"ok": True}


# --- Phase A: parse layer ---------------------------------------------------

def test_pdf_parser_extracts_text():
    from ingestion.parsers import pdf_parser

    doc = pdf_parser.parse(FIXTURES / "sample.pdf")
    assert "3503" in doc.text or "Hirth" in doc.text
    assert doc.metadata["type"] == "pdf"
    assert doc.metadata["pages"] >= 1
    assert isinstance(doc.images, list)
    assert doc.source_ref["filename"] == "sample.pdf"


def test_pdf_parser_skips_bad_pages(monkeypatch, tmp_path):
    from ingestion.parsers import pdf_parser

    class FakePage:
        def __init__(self, text="", fail=False):
            self.text = text
            self.fail = fail

        def get_text(self, *args, **kwargs):
            if self.fail:
                raise RuntimeError("bad page layout")
            return self.text

        def get_images(self, full=True):
            return []

    class FakeDoc:
        is_encrypted = False
        metadata = {"title": "fake"}

        def __init__(self):
            self.pages = [
                FakePage("Good page 1"),
                FakePage(fail=True),
                FakePage("Good page 3"),
            ]

        def __len__(self):
            return len(self.pages)

        def __getitem__(self, index):
            return self.pages[index]

        def close(self):
            pass

    monkeypatch.setattr(pdf_parser.fitz, "open", lambda path: FakeDoc())
    doc = pdf_parser.parse(tmp_path / "faulty.pdf")

    assert "Good page 1" in doc.text
    assert "Good page 3" in doc.text
    assert doc.metadata["failed_pages"] == [2]
    assert doc.source_ref["page_count"] == 3


def test_docx_parser_extracts_text_and_tables():
    from ingestion.parsers import docx_parser

    doc = docx_parser.parse(FIXTURES / "sample.docx")
    assert "Hirth" in doc.text or "manual" in doc.text.lower()
    assert doc.metadata["type"] == "docx"
    assert len(doc.tables) >= 1
    # first table should have a header row
    assert doc.tables[0].rows[0][0] != ""


def test_route_txt_returns_parseddoc(tmp_path):
    from ingestion.format_router import route

    f = tmp_path / "notes.txt"
    f.write_text("Hirth 3503 ignition timing spec.")
    doc = route(f)
    assert "Hirth" in doc.text
    assert doc.metadata["type"] == "text"
    assert "lang" in doc.metadata


def test_route_pdf_dispatches_to_pdf_parser():
    from ingestion.format_router import route

    doc = route(FIXTURES / "sample.pdf")
    assert doc.metadata["type"] == "pdf"
    assert "lang" in doc.metadata


def test_route_unsupported_raises():
    import pytest
    from ingestion.format_router import route

    with pytest.raises(ValueError):
        route(Path("file.xyz"))


# --- Phase B: normalize + chunk --------------------------------------------

def test_normalize_strips_boilerplate():
    from ingestion.corpus_builder import normalize
    from ingestion.types import ParsedDoc

    doc = ParsedDoc(
        text="Real content here.\n--------\nMore real content.\n\n\n\nEnd.",
        metadata={"filename": "test.txt", "type": "text"},
        source_ref={"filename": "test.txt"},
    )
    result = normalize(doc)
    assert "--------" not in result.text
    assert result.text.count("\n\n\n") == 0
    assert result.metadata["filename"] == "test.txt"


def test_normalize_fills_missing_metadata():
    from ingestion.corpus_builder import normalize
    from ingestion.types import ParsedDoc

    doc = ParsedDoc(text="Some text.", metadata={}, source_ref={})
    result = normalize(doc)
    assert result.metadata["filename"] == "unknown"
    assert result.metadata["type"] == "unknown"


def test_chunker_splits_prose():
    from ingestion.chunker import chunk
    from ingestion.types import ParsedDoc

    long_text = "word " * 500  # ~2500 chars
    doc = ParsedDoc(
        text=long_text,
        metadata={"filename": "test.txt", "type": "text", "lang": "en"},
        source_ref={"filename": "test.txt"},
    )
    chunks = chunk(doc, size=700, overlap=100)
    prose_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "prose"]
    assert len(prose_chunks) >= 3
    for c in prose_chunks:
        assert len(c["content"]) <= 850  # chunker may extend to sentence boundary (+150)
        ref = c["source_refs"][0]
        assert ref["filename"] == "test.txt"  # page key may also be present


def test_chunker_preserves_tables():
    from ingestion.chunker import chunk
    from ingestion.types import ParsedDoc, Table

    doc = ParsedDoc(
        text="Short prose.",
        tables=[Table(name="specs", rows=[["Param", "Value"], ["RPM", "6500"]])],
        metadata={"filename": "manual.pdf", "type": "pdf", "lang": "en"},
        source_ref={"filename": "manual.pdf"},
    )
    chunks = chunk(doc)
    table_chunks = [c for c in chunks if c["metadata"].get("chunk_type") == "table"]
    assert len(table_chunks) == 1
    assert "RPM" in table_chunks[0]["content"]
    assert "6500" in table_chunks[0]["content"]


# --- Phase C: embed --------------------------------------------------------

def test_embed_returns_correct_dim():
    from ingestion.knowledge_base import embed
    from config import get_settings

    chunks = [
        {"content": "Hirth 3503 ignition timing.", "metadata": {}, "source_refs": []},
        {"content": "Kraftstoff-Übersicht für Zweitaktmotoren.", "metadata": {}, "source_refs": []},
    ]
    result = embed(chunks)
    dim = get_settings().embedding_dim
    assert all("embedding" in c for c in result)
    assert all(len(c["embedding"]) == dim for c in result)


def test_embed_empty_list():
    from ingestion.knowledge_base import embed

    assert embed([]) == []


# --- Slice 2: sheet_parser, ocr, doc_convert, spec_lookup -------------------

def test_sheet_parser_csv(tmp_path):
    from ingestion.parsers import sheet_parser

    f = tmp_path / "specs.csv"
    f.write_text("Parameter,Value [rpm],Torque [Nm]\nIdle,800,5.2\nMax,6500,12.8\n")
    doc = sheet_parser.parse(f)
    assert doc.metadata["type"] == "sheet"
    assert doc.metadata["filename"] == "specs.csv"
    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.rows[0] == ["Parameter", "Value [rpm]", "Torque [Nm]"]
    assert table.units == {"Value [rpm]": "rpm", "Torque [Nm]": "Nm"}
    assert "800" in table.rows[1]
    assert "Parameter" in doc.text  # column names appear in prose summary


def test_sheet_parser_xlsx():
    from ingestion.parsers import sheet_parser

    doc = sheet_parser.parse(FIXTURES / "sample.xlsx")
    assert doc.metadata["type"] == "sheet"
    assert doc.metadata["filename"] == "sample.xlsx"
    assert len(doc.tables) >= 1
    table = doc.tables[0]
    assert len(table.rows) >= 2        # header + at least one data row
    assert table.units                 # bracket-notation units detected
    assert "6500" in str(table.rows)   # Max Power row present


def test_sheet_parser_units_flow_into_chunker():
    """Units stored in Table.units must appear in the chunk's Units: line."""
    from ingestion.parsers import sheet_parser
    from ingestion.chunker import chunk

    doc = sheet_parser.parse(FIXTURES / "sample.xlsx")
    chunks = chunk(doc)
    table_chunks = [c for c in chunks if c["metadata"].get("chunk_type") == "table"]
    assert table_chunks, "expected at least one table chunk"
    assert "Units:" in table_chunks[0]["content"]


def test_ocr_png():
    import shutil
    import pytest

    if not shutil.which("tesseract"):
        pytest.skip("tesseract not installed")

    from ingestion.parsers import ocr

    doc = ocr.parse(FIXTURES / "sample.png")
    assert doc.metadata["type"] == "ocr"
    assert doc.metadata["pages"] == 1
    assert isinstance(doc.metadata["low_confidence_pages"], list)
    assert doc.source_ref["filename"] == "sample.png"
    assert doc.source_ref["page_count"] == 1


def test_doc_convert_no_system_dep(tmp_path):
    """When neither libreoffice nor antiword is on PATH, raise RuntimeError."""
    import shutil
    import pytest
    from unittest.mock import patch

    if shutil.which("libreoffice") or shutil.which("antiword"):
        pytest.skip("system dep present; skipping no-dep error path")

    stub = tmp_path / "legacy.doc"
    stub.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")  # OLE2 magic bytes

    from ingestion.parsers import doc_convert

    with pytest.raises(RuntimeError, match="libreoffice"):
        doc_convert.parse(stub)


# --- Slice 3: reranker, verifier, graph -------------------------------------

def test_reranker_sorts_and_truncates():
    """Cross-encoder scores must sort results descending, capped at RERANK_TOP_K."""
    from unittest.mock import patch, MagicMock
    from config import get_settings

    candidates = [
        {"content": "Low relevance text", "doc_id": "a"},
        {"content": "Highly relevant RPM spec", "doc_id": "b"},
        {"content": "Medium relevance content", "doc_id": "c"},
        {"content": "Another low relevance doc", "doc_id": "d"},
        {"content": "Top match for the query", "doc_id": "e"},
    ]
    mock_scores = [0.1, 0.85, 0.5, 0.05, 0.95]
    mock_ce = MagicMock()
    mock_ce.predict.return_value = mock_scores

    with patch("agent.reranker._model", return_value=mock_ce):
        from agent.reranker import rerank
        results = rerank("RPM specification", candidates)

    top_k = get_settings().rerank_top_k
    assert len(results) <= top_k
    assert results[0]["doc_id"] == "e"   # score 0.95
    assert results[1]["doc_id"] == "b"   # score 0.85
    assert "rerank_score" in results[0]
    assert results[0]["rerank_score"] == pytest.approx(0.95, abs=1e-6)


def test_reranker_empty_input():
    from agent.reranker import rerank
    assert rerank("anything", []) == []


def test_verifier_grounded_no_numbers():
    """A draft with no numeric claims is always grounded."""
    from agent.verifier import is_grounded
    assert is_grounded("I don't have enough information to answer.", [])


def test_verifier_grounded_true():
    from agent.verifier import is_grounded
    ctx = [{"content": "The Hirth 3503 operates at a maximum of 6500 rpm."}]
    assert is_grounded("The engine reaches 6500 rpm at full power [Source 1].", ctx)


def test_verifier_grounded_false():
    from agent.verifier import is_grounded
    ctx = [{"content": "The engine produces 12.8 Nm of torque."}]
    # 7000 does not appear anywhere in the source
    assert not is_grounded("The engine produces 7000 rpm [Source 1].", ctx)


def test_verifier_grounded_decimal():
    from agent.verifier import is_grounded
    ctx = [{"content": "Torque at peak: 12.8 Nm"}]
    assert is_grounded("Peak torque is 12.8 Nm [Source 1].", ctx)


def test_graph_answer_returns_expected_shape():
    """Full graph round-trip with mocked LLM: shape and types must match API contract."""
    import importlib
    from unittest.mock import patch

    # chat_json drives reason() (returns "answer" immediately) then verify()'s
    # related-questions call
    chat_json_calls = iter([
        {"thought": "Answering directly", "action": "answer", "args": {}},
        ["What is the idle RPM?", "How to check ignition timing?"],
    ])

    def fake_chat_json(messages, **kw):
        return next(chat_json_calls)

    def fake_chat(messages, **kw):
        return "I don't have enough information in the knowledge base to answer that question."

    import agent.graph as ag
    ag._graph = None  # reset singleton so tests don't bleed

    with patch("llm.chat_json", side_effect=fake_chat_json), \
         patch("llm.chat", side_effect=fake_chat):
        result = ag.answer("What is the ignition timing for the Hirth 3503?")

    assert isinstance(result["answer"], str)
    assert isinstance(result["citations"], list)
    assert result["confidence"] in ("high", "low")
    assert isinstance(result["related_questions"], list)


def test_spec_lookup_queries_db():
    """spec_lookup issues the right SQL and maps rows to dicts with source_ref."""
    from unittest.mock import MagicMock, patch

    mock_row = (
        "fuel_data", "Specs", "Benzin", "Value [rpm]",
        "Benzin::Value [rpm]", "6500", "rpm", {"filename": "fuel_data.xlsx"},
    )
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_cur.fetchall.return_value = [mock_row]

    with patch("config.get_connection", return_value=mock_conn):
        from agent import tools
        # reload to pick up patched get_connection
        import importlib; importlib.reload(tools)
        results = tools.spec_lookup("Value [rpm]")

    assert len(results) == 1
    r = results[0]
    assert r["value"] == "6500"
    assert r["unit"] == "rpm"
    assert r["key"] == "Benzin::Value [rpm]"
    assert r["source_ref"] == {"filename": "fuel_data.xlsx"}


# --- Slice 4: recommender + reranker wired into act() -----------------------

def test_recommender_llm_path():
    """When the graph DB is unavailable, recommender falls back to LLM suggestions."""
    from unittest.mock import patch

    chunks = [{"content": "Hirth 3503 ignition timing is 28 degrees BTDC."}]

    def fake_chat_json(messages, **kw):
        return ["What is the idle RPM?", "How to adjust ignition timing?"]

    with patch("llm.chat_json", side_effect=fake_chat_json), \
         patch("config.get_connection", side_effect=RuntimeError("no db")):
        from agent.recommender import related
        result = related("What is the ignition timing?", chunks)

    assert isinstance(result, list)
    assert 1 <= len(result) <= 3
    assert all(isinstance(q, str) for q in result)


def test_recommender_empty_chunks():
    """No chunks and no graph → empty list, no error."""
    from unittest.mock import patch

    with patch("config.get_connection", side_effect=RuntimeError("no db")):
        from agent.recommender import related
        result = related("What is the RPM?", [])

    assert result == []


def test_act_reranks_hybrid_search_results():
    """act() must apply the reranker after hybrid_search and return results in reranked order."""
    from unittest.mock import patch, MagicMock
    from agent.nodes import act, AgentState

    raw = [
        {"content": "Low relevance text", "doc_id": "a", "source_refs": [], "metadata": {}, "score": 0.5},
        {"content": "High relevance text", "doc_id": "b", "source_refs": [], "metadata": {}, "score": 0.4},
    ]
    reranked = [
        {**raw[1], "rerank_score": 0.92},
        {**raw[0], "rerank_score": 0.11},
    ]
    mock_rerank = MagicMock(return_value=reranked)

    state: AgentState = {
        "question": "What is the max RPM?",
        "lang": "en",
        "expertise": "expert",
        "scratch": [],
        "tool_action": "hybrid_search",
        "tool_args": {"query": "max RPM"},
        "draft": "",
        "citations": [],
        "grounded": False,
        "loops": 1,
        "related": [],
    }

    with patch("agent.retriever_hybrid.search", return_value=raw), \
         patch("agent.reranker.rerank", mock_rerank):
        result = act(state)

    assert result["scratch"][0]["results"][0]["doc_id"] == "b"  # highest rerank score first
    mock_rerank.assert_called_once_with("max RPM", raw)


# --- Slice 5: memory, dedup, enricher, graph_builder, tools -----------------

def test_memory_record_feedback():
    """record_feedback executes an INSERT on the feedback table."""
    from unittest.mock import MagicMock, patch

    mock_conn = MagicMock()
    mock_cur  = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    with patch("config.get_connection", return_value=mock_conn):
        from memory.store import record_feedback
        record_feedback("sess1", "What is RPM?", "It is 6500.", vote=1)

    sqls = [call.args[0] for call in mock_cur.execute.call_args_list]
    assert any("INSERT INTO feedback" in sql for sql in sqls)


def test_memory_record_feedback_updates_chunk_quality_scores():
    """Votes tied to cited chunks update quality_score counters."""
    from unittest.mock import MagicMock, patch

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    with patch("config.get_connection", return_value=mock_conn):
        from memory.store import record_feedback
        record_feedback("sess1", "What is RPM?", "Answer", vote=-1, chunk_ids=[42])

    sqls = [call.args[0] for call in mock_cur.execute.call_args_list]
    assert any("INSERT INTO feedback" in sql for sql in sqls)
    assert any("downvote_count = downvote_count + 1" in sql for sql in sqls)
    assert any(call.args[1] == (42,) for call in mock_cur.execute.call_args_list if len(call.args) > 1)


def test_memory_record_feedback_stores_long_correction_as_trusted_chunk():
    """Expert corrections become high-quality chunks for future retrieval."""
    from unittest.mock import MagicMock, patch

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = [[0.1] * 384]

    with patch("config.get_connection", return_value=mock_conn), \
         patch("ingestion.knowledge_base._embedder", return_value=mock_embedder):
        from memory.store import record_feedback
        record_feedback(
            "sess1",
            "What is the temperature?",
            "Old answer",
            vote=-1,
            correction="The actual verified value is 215 C, not 220 C.",
        )

    sqls = [call.args[0] for call in mock_cur.execute.call_args_list]
    assert any("INSERT INTO chunks" in sql and "quality_score" in sql for sql in sqls)


def test_memory_get_conversation_empty():
    """get_conversation returns [] for an unknown session."""
    from unittest.mock import MagicMock, patch

    mock_conn = MagicMock()
    mock_cur  = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn.cursor.return_value  = mock_cur

    with patch("config.get_connection", return_value=mock_conn):
        from memory.store import get_conversation
        turns = get_conversation("unknown-session")

    assert turns == []


def test_dedup_skips_near_duplicate():
    """dedup_and_merge drops a chunk when a near-identical vector is already stored."""
    from unittest.mock import MagicMock, patch

    chunk = {
        "content": "The max RPM is 6500.",
        "embedding": [0.1] * 384,
        "metadata": {},
        "source_refs": [{"filename": "manual.pdf"}],
    }

    # _find_near_duplicate returns id=42 → chunk is a duplicate
    with patch("ingestion.dedup._find_near_duplicate", return_value=42), \
         patch("ingestion.dedup._merge_source_refs") as mock_merge:
        from ingestion.dedup import dedup_and_merge
        result = dedup_and_merge([chunk])

    assert result == []
    mock_merge.assert_called_once_with(42, [{"filename": "manual.pdf"}])


def test_dedup_keeps_unique_chunk():
    """dedup_and_merge keeps a chunk when no near-duplicate exists."""
    from unittest.mock import patch

    chunk = {
        "content": "Unique spec text.",
        "embedding": [0.1] * 384,
        "metadata": {},
        "source_refs": [],
    }

    with patch("ingestion.dedup._find_near_duplicate", return_value=None):
        from ingestion.dedup import dedup_and_merge
        result = dedup_and_merge([chunk])

    assert len(result) == 1


def test_domain_enricher_attaches_entities():
    """enrich() attaches entities and tags to chunk metadata on LLM success."""
    from unittest.mock import patch

    chunks = [{"content": "A" * 100, "metadata": {"filename": "x.pdf"}, "source_refs": []}]

    def fake_chat_json(messages, **kw):
        return {"entities": [{"type": "engine", "name": "Hirth 3503"}], "tags": ["ignition"]}

    with patch("llm.chat_json", side_effect=fake_chat_json):
        from ingestion.domain_enricher import enrich
        result = enrich(chunks)

    assert result[0]["metadata"]["entities"] == [{"type": "engine", "name": "Hirth 3503"}]
    assert "ignition" in result[0]["metadata"]["tags"]


def test_domain_enricher_skips_short_chunks():
    """Chunks shorter than the minimum threshold are returned unchanged."""
    from ingestion.domain_enricher import enrich

    chunk = {"content": "Short.", "metadata": {}, "source_refs": []}
    result = enrich([chunk])
    assert "entities" not in result[0].get("metadata", {})


def test_unit_convert_basic():
    from agent.tools import unit_convert
    assert unit_convert(1.0, "kW", "hp") == pytest.approx(1.34102, rel=1e-3)
    assert unit_convert(100.0, "°C", "°F") == pytest.approx(212.0, rel=1e-3)
    assert unit_convert(0.0, "°C", "°F") == pytest.approx(32.0, rel=1e-3)
    assert unit_convert(6500.0, "rpm", "rpm") == pytest.approx(6500.0)


def test_unit_convert_unknown_raises():
    from agent.tools import unit_convert
    with pytest.raises(ValueError, match="no conversion"):
        unit_convert(1.0, "furlong", "fortnight")


def test_conflict_check_flags_divergence():
    """conflict_check marks rows as conflict=True when multiple values exist for a key."""
    from unittest.mock import MagicMock, patch

    rows = [
        ("max_rpm::Value [rpm]", "6500", "doc_a", {}),
        ("max_rpm::Value [rpm]", "7000", "doc_b", {}),
    ]
    mock_conn = MagicMock()
    mock_cur  = MagicMock()
    mock_cur.fetchall.return_value = rows
    mock_conn.cursor.return_value  = mock_cur

    with patch("config.get_connection", return_value=mock_conn):
        from agent.tools import conflict_check
        result = conflict_check("max_rpm")

    assert len(result) == 2
    assert all(r["conflict"] for r in result)


def test_source_viewer_not_found():
    """source_viewer returns an error dict when chunk_id does not exist."""
    from unittest.mock import MagicMock, patch

    mock_conn = MagicMock()
    mock_cur  = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn.cursor.return_value  = mock_cur

    with patch("config.get_connection", return_value=mock_conn):
        from agent.tools import source_viewer
        result = source_viewer(9999)

    assert "error" in result


# --- Slice 6: graph_builder, graph_lookup, diagnostic_tree, expertise mode --

def test_kg_ontology_seed_graph_is_valid_and_repeatable():
    """Manual seed graph has valid ontology types and stable node identity."""
    from ingestion.kg_ontology import EDGE_TYPES, NODE_TYPES, is_valid_seed_graph, seed_graph

    first = seed_graph()
    second = seed_graph()

    assert is_valid_seed_graph(first)
    assert first == second
    assert any(n["type"] == "system" and n["name"] == "Ignition System" for n in first["nodes"])
    assert any(n["type"] == "part" and n["name"] == "Spark Plug" for n in first["nodes"])
    assert any(n["type"] == "symptom" and n["name"] == "Won't Start" for n in first["nodes"])
    assert all(n["type"] in NODE_TYPES for n in first["nodes"])
    assert all(e["relation"] in EDGE_TYPES for e in first["edges"])


def test_kg_normalizer_canonicalizes_synonyms_and_rejects_noise():
    """Common Hirth/two-stroke terms merge; obvious document noise is rejected."""
    from ingestion.kg_normalizer import classify_entity_type, normalize_entity

    assert not normalize_entity("July 2013", "engine")["is_valid"]
    assert not normalize_entity("p. 12", "engine")["is_valid"]

    for raw in ("spark plug", "plug", "Zündkerze"):
        result = normalize_entity(raw, "part")
        assert result["is_valid"]
        assert result["canonical_name"] == "Spark Plug"
        assert result["type"] == "part"

    for raw in ("carburettor", "Vergaser"):
        result = normalize_entity(raw, "part")
        assert result["is_valid"]
        assert result["canonical_name"] == "Carburetor"
        assert result["type"] == "part"

    assert normalize_entity("700°C", "part")["type"] == "spec"
    assert normalize_entity("150 hours", "engine")["type"] == "spec"
    assert normalize_entity("Hirth 3503", "engine")["type"] == "engine"
    assert classify_entity_type("Hirth 3503") == "engine"


def test_graph_builder_calls_llm_and_upserts():
    """graph_builder.extract() calls the LLM and attempts DB upsert."""
    from unittest.mock import MagicMock, patch
    from ingestion.types import ParsedDoc

    doc = ParsedDoc(
        text="The Hirth 3503 engine overheats when the carburettor jets are clogged.",
        metadata={"filename": "manual.pdf", "type": "pdf"},
        source_ref={"filename": "manual.pdf"},
    )
    llm_result = {
        "nodes": [
            {"type": "engine",  "name": "Hirth 3503"},
            {"type": "symptom", "name": "overheating"},
            {"type": "cause",   "name": "clogged jets"},
        ],
        "edges": [
            {"src": "Hirth 3503", "dst": "overheating", "relation": "has_symptom"},
        ],
    }

    mock_conn = MagicMock()
    mock_cur  = MagicMock()
    mock_cur.fetchone.return_value = (1,)
    mock_conn.cursor.return_value  = mock_cur

    with patch("llm.chat_json", return_value=llm_result), \
         patch("config.get_connection", return_value=mock_conn):
        from ingestion import graph_builder
        graph_builder.extract(doc)

    # Should have called execute for node inserts + edge insert
    assert mock_cur.execute.call_count >= 2


def test_graph_builder_uses_later_chunks_for_kg_extraction():
    """KG extraction uses selected chunks, not only the document prefix."""
    from unittest.mock import patch
    from ingestion.types import ParsedDoc

    doc = ParsedDoc(
        text="A" * 2000,
        metadata={"filename": "manual.pdf", "type": "pdf"},
        source_ref={"filename": "manual.pdf"},
    )
    chunks = [
        {
            "content": "A" * 700,
            "metadata": {"filename": "manual.pdf", "chunk_index": 0, "page": 1},
            "source_refs": [{"filename": "manual.pdf", "page": 1}],
        },
        {
            "content": "Troubleshooting: check the spark plug if Hirth 3503 misfires at 700°C.",
            "metadata": {"filename": "manual.pdf", "chunk_index": 7, "page": 9},
            "source_refs": [{"filename": "manual.pdf", "page": 9}],
        },
    ]
    captured = {"nodes": [], "edges": []}

    def fake_upsert(nodes, edges):
        captured["nodes"].extend(nodes)
        captured["edges"].extend(edges)
        return {
            "nodes_inserted": len(nodes),
            "edges_inserted": len(edges),
            "nodes_rejected": 0,
            "edges_rejected": 0,
        }

    with patch("ingestion.graph_builder._seed", return_value={
        "nodes_inserted": 0,
        "edges_inserted": 0,
        "nodes_rejected": 0,
        "edges_rejected": 0,
    }), patch("ingestion.graph_builder._upsert", side_effect=fake_upsert):
        from ingestion import graph_builder
        stats = graph_builder.extract(doc, chunks=chunks)

    assert stats["chunks_considered"] == 2
    assert stats["chunks_selected"] == 2
    assert any(n["name"] == "Spark Plug" for n in captured["nodes"])
    assert any(n["type"] == "spec" and "700" in n["name"] for n in captured["nodes"])
    assert any(e["props"].get("source_chunk_id") == 7 for e in captured["edges"])


def test_graph_edge_props_are_validated_and_short():
    """Graph edge provenance keeps evidence short and validates confidence/method."""
    from ingestion.graph_builder import make_edge_props

    props = make_edge_props(
        doc_id="manual.pdf",
        chunk_id=12,
        page=3,
        evidence=(" Spark plug evidence. " * 40),
        confidence=3.5,
        extraction_method="not-a-method",
        extra={"source_title": "Manual"},
    )

    assert props["doc_id"] == "manual.pdf"
    assert props["source_chunk_id"] == 12
    assert props["chunk_id"] == 12
    assert props["page"] == 3
    assert props["confidence"] == 1.0
    assert props["extraction_method"] == "unknown"
    assert props["source_title"] == "Manual"
    assert isinstance(props["evidence"], str)
    assert len(props["evidence"]) <= 300


def test_graph_serializers_preserve_backward_compatibility():
    from api.main import serialize_graph_edge, serialize_graph_node

    node = serialize_graph_node(1, "part", "spark plug", {"aliases": ["plug"], "doc_id": "manual.pdf"})
    edge = serialize_graph_edge(10, 1, 2, "HAS_SPEC", {})

    assert node["id"] == 1
    assert node["name"] == "Spark Plug"
    assert node["label"] == "Spark Plug"
    assert node["canonical_name"] == "Spark Plug"
    assert node["doc_ids"] == ["manual.pdf"]
    assert "props" in node

    assert edge["id"] == 10
    assert edge["source"] == 1
    assert edge["target"] == 2
    assert edge["relation"] == "HAS_SPEC"
    assert edge["type"] == "HAS_SPEC"
    assert edge["confidence"] is None
    assert edge["extraction_method"] == "unknown"
    assert edge["props"]["evidence"] == ""


def test_api_graph_endpoint_supports_filters_and_rich_metadata():
    from fastapi.testclient import TestClient
    from unittest.mock import MagicMock, patch
    from api.main import app

    node_rows = [
        (1, "part", "Spark Plug", {"doc_id": "manual.pdf", "aliases": ["plug"], "confidence": 0.8}),
        (2, "spec", "700°C", {"doc_id": "manual.pdf", "confidence": 0.8}),
        (3, "engine", "July 2013", {}),
    ]
    edge_rows = [
        (10, 1, 2, "HAS_SPEC", {
            "doc_id": "manual.pdf",
            "source_chunk_id": 4,
            "page": 12,
            "evidence": "Spark plug temperature 700°C.",
            "confidence": 0.82,
            "extraction_method": "rule",
            "source_title": "Manual",
            "topic": "Drehmomente",
            "relative_path": "17. Drehmomente/manual.pdf",
            "file_type": "pdf",
        }),
        (11, 1, 2, "RELATED_TO", {
            "doc_id": "manual.pdf",
            "confidence": 0.2,
            "extraction_method": "rule",
        }),
    ]

    mock_cur = MagicMock()
    mock_cur.fetchall.side_effect = [node_rows, edge_rows]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    with patch("config.get_connection", return_value=mock_conn):
        client = TestClient(app)
        resp = client.get(
            "/graph?edge_type=HAS_SPEC&doc_id=manual.pdf&min_confidence=0.5"
            "&extraction_method=rule&include_seed=false&topic=Drehmomente"
        )

    data = resp.json()
    assert resp.status_code == 200
    assert len(data["edges"]) == 1
    edge = data["edges"][0]
    assert edge["id"] == 10
    assert edge["confidence"] == 0.82
    assert edge["evidence"] == "Spark plug temperature 700°C."
    assert edge["source_chunk_id"] == 4
    assert edge["source_title"] == "Manual"
    assert edge["topic"] == "Drehmomente"
    assert edge["relative_path"] == "17. Drehmomente/manual.pdf"
    assert data["nodes"][0]["label"] == "Spark Plug"
    assert data["filters"]["edge_type"] == "HAS_SPEC"
    assert data["filters"]["topic"] == "Drehmomente"


def test_graph_diagnostic_paths_endpoint():
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from api.main import app

    fake = {
        "intent": {"intent": "diagnostic_cause"},
        "paths": [{"path": "Misfire --CAUSED_BY--> Spark Plug Fouling"}],
        "context": "Knowledge Graph evidence:",
    }
    with patch("agent.kg_retrieval.retrieve_kg_context", return_value=fake):
        client = TestClient(app)
        resp = client.get("/graph/diagnostic-paths?query=misfire")

    data = resp.json()
    assert resp.status_code == 200
    assert data["query"] == "misfire"
    assert data["paths"][0]["path"].startswith("Misfire")


def test_kg_retrieval_intent_and_context_formatting():
    from agent.kg_retrieval import classify_question_intent, format_kg_context

    assert classify_question_intent("Why does the engine misfire?")["intent"] == "diagnostic_cause"
    assert classify_question_intent("What should I check for overheating?")["intent"] == "diagnostic_fix"
    assert classify_question_intent("What is the spark plug related to?")["intent"] == "part_lookup"
    assert classify_question_intent("Tell me about the company")["intent"] == "general_question"

    context = format_kg_context([
        {
            "path": "Misfire --CAUSED_BY--> Spark Plug Fouling --FIXED_BY--> Replace Spark Plug",
            "evidence": "Engine misfire may be caused by spark plug fouling.",
            "source_title": "Manual",
            "page": 12,
            "source_chunk_id": 4,
            "confidence": 0.86,
        }
    ])
    assert "Knowledge Graph evidence" in context
    assert "Misfire --CAUSED_BY-->" in context
    assert "Manual, page 12, chunk 4" in context


def test_kg_rules_extract_diagnostic_cause_and_fixes():
    """Rule extraction works without an LLM for diagnostic snippets."""
    from ingestion.graph_builder import _chunk_props
    from ingestion.kg_rules import extract_rules

    chunk = {
        "content": "Engine misfire may be caused by spark plug fouling. Clean or replace the spark plug.",
        "metadata": {"filename": "manual.pdf", "chunk_index": 4, "page": 8},
        "source_refs": [{"filename": "manual.pdf", "page": 8}],
    }

    result = extract_rules(chunk, _chunk_props)
    nodes = {(n["type"], n["name"]) for n in result["nodes"]}
    edges = {(e["src"], e["relation"], e["dst"]) for e in result["edges"]}

    assert ("symptom", "Misfire") in nodes
    assert ("cause", "Spark Plug Fouling") in nodes
    assert ("fix", "Clean Spark Plug") in nodes
    assert ("fix", "Replace Spark Plug") in nodes
    assert ("part", "Spark Plug") in nodes
    assert ("Misfire", "CAUSED_BY", "Spark Plug Fouling") in edges
    assert ("Spark Plug Fouling", "FIXED_BY", "Clean Spark Plug") in edges
    assert ("Spark Plug Fouling", "FIXED_BY", "Replace Spark Plug") in edges
    assert all(e["props"]["source_chunk_id"] == 4 for e in result["edges"])
    assert all(e["props"]["extraction_method"] == "rule" for e in result["edges"])


def test_kg_rules_extract_table_specs_and_maintenance():
    from ingestion.graph_builder import _chunk_props
    from ingestion.kg_rules import extract_rules

    chunk = {
        "content": (
            "symptom | possible cause | correction\n"
            "rough idle | blocked jet | clean carburetor\n"
            "Check carburetor idle speed 2200 rpm every 150 hours. "
            "Warning: do not operate with damaged fuel line."
        ),
        "metadata": {"filename": "manual.pdf", "chunk_index": 9, "page": 12, "chunk_type": "table"},
        "source_refs": [{"filename": "manual.pdf", "page": 12}],
    }

    result = extract_rules(chunk, _chunk_props)
    nodes = {(n["type"], n["name"]) for n in result["nodes"]}
    edges = {(e["src"], e["relation"], e["dst"]) for e in result["edges"]}

    assert ("symptom", "Rough Idle") in nodes
    assert ("cause", "Blocked Jet") in nodes
    assert ("fix", "Clean Carburetor") in nodes
    assert any(node[0] == "spec" and "2200" in node[1] for node in nodes)
    assert any(node[0] == "spec" and "150" in node[1] for node in nodes)
    assert ("Rough Idle", "CAUSED_BY", "Blocked Jet") in edges
    assert ("Blocked Jet", "FIXED_BY", "Clean Carburetor") in edges
    assert any(e["relation"] == "HAS_SPEC" for e in result["edges"])
    assert any(e["relation"] == "REQUIRES_PROCEDURE" for e in result["edges"])
    assert any(e["relation"] == "RELATED_TO" for e in result["edges"])
    assert all(e["props"]["extraction_method"] in ("rule", "table") for e in result["edges"])


def test_graph_answer_beginner_mode():
    """graph.answer() with expertise='beginner' produces a result with the right shape."""
    import importlib
    from unittest.mock import patch

    chat_json_calls = iter([
        {"thought": "Answering directly", "action": "answer", "args": {}},
        ["Follow-up question?"],
    ])

    def fake_chat_json(messages, **kw):
        return next(chat_json_calls)

    def fake_chat(messages, **kw):
        # Verify the beginner style note is present in the system prompt
        system = messages[0]["content"]
        assert "plain" in system.lower() or "beginner" in system.lower() or "jargon" in system.lower()
        return "I don't have enough information to answer that."

    import agent.graph as ag
    ag._graph = None

    with patch("llm.chat_json", side_effect=fake_chat_json), \
         patch("llm.chat", side_effect=fake_chat):
        result = ag.answer("What is ignition timing?", expertise="beginner")

    assert isinstance(result["answer"], str)
    assert result["confidence"] in ("high", "low")


def test_api_health_endpoint():
    """Health endpoint returns ok; the removed /gaps endpoint now 404s."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    assert client.get("/health").json() == {"ok": True}

    # /gaps was removed with the Knowledge Gaps feature
    assert client.get("/gaps").status_code == 404


def test_api_graph_endpoint():
    """/graph returns nodes and edges keys."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    data = client.get("/graph").json()
    assert "nodes" in data
    assert "edges" in data


# --- Slice 7: pptx_parser, BM25/RRF, figure_handler, stream_chat, versioning --

def test_pptx_parser_extracts_slides():
    """pptx_parser returns slide text from every shape and speaker notes."""
    from ingestion.parsers import pptx_parser

    doc = pptx_parser.parse(FIXTURES / "sample.pptx")
    assert doc.metadata["type"] == "pptx"
    assert doc.metadata["slides"] == 2
    assert isinstance(doc.images, list)
    assert "Hirth 3503" in doc.text
    assert "6500" in doc.text
    assert "Speaker notes" in doc.text or "2-stroke oil" in doc.text
    assert "--- Slide 1 ---" in doc.text
    assert "--- Slide 2 ---" in doc.text


def test_pptx_parser_source_ref():
    """source_ref must carry filename and slide_count."""
    from ingestion.parsers import pptx_parser

    doc = pptx_parser.parse(FIXTURES / "sample.pptx")
    assert doc.source_ref["filename"] == "sample.pptx"
    assert doc.source_ref["slide_count"] == 2


def test_format_router_routes_pptx():
    """format_router.route() dispatches .pptx to pptx_parser without error."""
    from ingestion.format_router import route

    doc = route(FIXTURES / "sample.pptx")
    assert doc.metadata["type"] == "pptx"
    assert "lang" in doc.metadata


def test_rrf_fuse_combines_lists():
    """_rrf_fuse gives a chunk appearing in both lists a higher score than one in only one."""
    from agent.retriever_hybrid import _rrf_fuse

    chunk_both   = {"id": 1, "content": "in both",   "score": 0.9}
    chunk_dense  = {"id": 2, "content": "dense only", "score": 0.8}
    chunk_sparse = {"id": 3, "content": "sparse only","score": 0.7}

    dense  = [chunk_both, chunk_dense]
    sparse = [chunk_both, chunk_sparse]

    result = _rrf_fuse(dense, sparse, top_k=3)

    ids = [c["id"] for c in result]
    assert ids[0] == 1, "chunk in both lists should rank first"
    assert len(result) == 3


def test_rrf_fuse_top_k_limits_output():
    """_rrf_fuse must honour top_k even when inputs are larger."""
    from agent.retriever_hybrid import _rrf_fuse

    dense  = [{"id": i, "content": f"d{i}", "score": 1/(i+1)} for i in range(10)]
    sparse = [{"id": i+5, "content": f"s{i}", "score": 1/(i+1)} for i in range(10)]

    result = _rrf_fuse(dense, sparse, top_k=4)
    assert len(result) == 4


def test_rrf_fuse_suppresses_repeatedly_downvoted_chunks():
    from agent.retriever_hybrid import _rrf_fuse

    bad = {
        "id": 1,
        "content": "outdated spec",
        "score": 0.9,
        "quality_score": -0.8,
        "downvote_count": 3,
    }
    good = {
        "id": 2,
        "content": "trusted spec",
        "score": 0.5,
        "quality_score": 0.5,
        "downvote_count": 0,
    }

    result = _rrf_fuse([bad, good], [bad, good], top_k=2)
    assert [c["id"] for c in result] == [2]
    assert result[0]["score"] > 0


def test_conflict_detector_flags_cross_document_numeric_disagreement():
    from agent.conflict_detector import detect_conflicts, group_by_doc

    chunks = [
        {
            "id": 1,
            "doc_id": "spec.pdf",
            "content": "Maximum cylinder head temperature is 220 °C.",
            "metadata": {"source_title": "Spec Sheet", "page": 4},
        },
        {
            "id": 2,
            "doc_id": "manual.pdf",
            "content": "Cylinder head temperature limit: 210 °C during operation.",
            "metadata": {"source_title": "Maintenance Manual", "page": 8},
        },
    ]

    conflicts = detect_conflicts(group_by_doc(chunks), "max cylinder head temperature")

    assert len(conflicts) == 1
    assert conflicts[0]["severity"] == "conflict"
    assert {v["value"] for v in conflicts[0]["values"]} >= {"220", "210"}


def test_conflict_detector_ignores_years_and_non_spec_temperatures():
    from agent.conflict_detector import detect_conflicts, group_by_doc

    chunks = [
        {
            "id": 1,
            "doc_id": "slides.pdf",
            "content": "Presented in 2013 C section. --- Slide 4 --- No limit is stated.",
            "metadata": {"source_title": "Slides"},
        },
        {
            "id": 2,
            "doc_id": "notes.pdf",
            "content": "Page 4 C contains background text without a design value.",
            "metadata": {"source_title": "Notes"},
        },
    ]

    assert detect_conflicts(group_by_doc(chunks), "temperature") == []


def test_bm25_tokenize():
    """_tokenize lowercases and splits on whitespace — DE+EN corpus, no NLTK needed."""
    from agent.retriever_hybrid import _tokenize

    tokens = _tokenize("Hirth 3503 Zündung BTDC")
    assert tokens == ["hirth", "3503", "zündung", "btdc"]


def test_figure_handler_disabled_returns_empty():
    """With ENABLE_FIGURE_HANDLER=false, extract_and_caption returns an empty ParsedDoc."""
    from unittest.mock import patch
    from ingestion.parsers import figure_handler

    with patch("config.get_settings") as mock_settings:
        mock_settings.return_value.enable_figure_handler = False
        doc = figure_handler.extract_and_caption(FIXTURES / "sample.pdf")

    assert doc.text == ""
    assert doc.metadata["type"] == "figure"
    assert doc.images == []


def test_stream_chat_yields_tokens():
    """stream_chat must yield individual string tokens from the provider."""
    from unittest.mock import patch, MagicMock

    # Simulate OpenAI streaming: a list of chunk objects with delta.content
    def _make_chunk(text):
        c = MagicMock()
        c.choices[0].delta.content = text
        return c

    fake_stream = [_make_chunk("The "), _make_chunk("max "), _make_chunk("RPM "), _make_chunk("is 6500.")]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter(fake_stream)

    with patch("config.get_settings") as mock_cfg, \
         patch("openai.OpenAI", return_value=mock_client):
        mock_cfg.return_value.llm_provider = "openai"
        mock_cfg.return_value.llm_model    = "gpt-4o-mini"
        mock_cfg.return_value.openai_api_key = "test"

        from llm import stream_chat
        tokens = list(stream_chat([{"role": "user", "content": "What is max RPM?"}]))

    assert tokens == ["The ", "max ", "RPM ", "is 6500."]
    assert "".join(tokens) == "The max RPM is 6500."


def test_ask_stream_endpoint_sse_format():
    """/ask/stream emits well-formed SSE events ending with a 'done' event."""
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from api.main import app

    fake_tokens = iter(["The ", "answer ", "is 6500."])

    def fake_search(q, k=None): return []  # no chunks → straight to done event

    with patch("agent.retriever_hybrid.search", side_effect=fake_search):
        client = TestClient(app)
        resp = client.post(
            "/ask/stream",
            data={"question": "What is the max RPM?", "session_id": "test", "expertise": "expert"},
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n\n")
    events = []
    for line in lines:
        if line.startswith("data: "):
            import json as _json
            events.append(_json.loads(line[6:]))

    types = [e["type"] for e in events]
    assert "done" in types
    done = next(e for e in events if e["type"] == "done")
    assert "citations" in done
    assert "confidence" in done
    assert "related_questions" in done


def test_chunk_images_endpoint_shape():
    """Chunk image endpoint returns web URLs and captions for linked figures."""
    from fastapi.testclient import TestClient
    from unittest.mock import MagicMock, patch
    from api.main import app

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_cur.fetchall.return_value = [
        ("/images/manual_page2_img1.png", "data/images/manual_page2_img1.png", "Figure 2: Piston diagram", 2),
    ]

    with patch("config.get_connection", return_value=mock_conn):
        client = TestClient(app)
        resp = client.get("/chunks/42/images")

    assert resp.status_code == 200
    data = resp.json()
    assert data == [{
        "url": "/images/manual_page2_img1.png",
        "caption": "Figure 2: Piston diagram",
        "page_or_slide": 2,
    }]


def test_chunk_images_endpoint_falls_back_to_same_page():
    """If a citation chunk is not directly linked, return images from the same doc/page."""
    from fastapi.testclient import TestClient
    from unittest.mock import MagicMock, patch
    from api.main import app

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_cur.fetchall.side_effect = [
        [],
        [("/images/manual_page8_img1.png", "data/images/manual_page8_img1.png", "Bild 1: combustion chamber", 8)],
    ]
    mock_cur.fetchone.return_value = ("manual.pdf", "8")

    with patch("config.get_connection", return_value=mock_conn):
        client = TestClient(app)
        resp = client.get("/chunks/99/images")

    assert resp.status_code == 200
    assert resp.json()[0]["url"] == "/images/manual_page8_img1.png"


def test_fetch_images_for_citations_returns_answer_image_payload():
    from unittest.mock import MagicMock, patch
    from api.main import _fetch_images_for_citations

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_cur.fetchall.return_value = [
        ("/images/manual_page3_img1.png", "data/images/manual_page3_img1.png", "Figure 3", 3),
    ]

    with patch("config.get_connection", return_value=mock_conn):
        images = _fetch_images_for_citations([
            {"n": 1, "doc_id": "manual.pdf", "page": 3},
        ])

    assert images == [{
        "url": "/images/manual_page3_img1.png",
        "caption": "Figure 3",
        "page": 3,
        "doc_id": "manual.pdf",
        "citation_n": 1,
    }]


def test_formula_extractor_detects_inline_and_latex():
    from ingestion.formula_extractor import extract_formulas

    text = (
        "Thermal efficiency is useful for engine comparison. "
        "eta = W_out / Q_in. W_out is work output and Q_in is heat input. "
        "A document may also write \\frac{W_{out}}{Q_{in}}."
    )
    formulas = extract_formulas(text)
    values = [f["formula_text"] for f in formulas]

    assert any("eta =" in v for v in values)
    assert any("\\frac" in v for v in values)
    assert any("work output" in f["context_after"] for f in formulas)


def test_math_normalizer_cleans_pdf_math_unicode():
    from ingestion.parsers.math_normalizer import normalize_math_unicode

    raw = (
        "\U0001D401\U0001D413\U0001D404 "
        "(\U0001D702\U0001D41E) = "
        "\U0001D48E \U0001D439\u2219\U0001D43F\U0001D43B\U0001D449 "
        "\U0001D443\U0001D483"
    )

    assert normalize_math_unicode(raw) == "BTE (eta_e) = m_dot_F * LHV / P_b"


def test_formula_search_endpoint_shape():
    from fastapi.testclient import TestClient
    from unittest.mock import MagicMock, patch
    from api.main import app

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_cur.fetchall.return_value = [
        (1, "manual.pdf", 42, "eta = W_out / Q_in", None, "Thermal efficiency.", "W_out is work.", 7, "Verbrennungsmotoren", "Manual"),
    ]

    with patch("config.get_connection", return_value=mock_conn):
        client = TestClient(app)
        resp = client.get("/formulas/search?q=thermal")

    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["formula_text"] == "eta = W_out / Q_in"
    assert data[0]["chunk_id"] == 42
    assert data[0]["page_or_slide"] == 7


def test_formula_core_term_strips_question_words():
    from api.main import _formula_core_term

    assert _formula_core_term("What is the thermal efficiency formula?") == "thermal efficiency"


def test_formula_mode_prefers_formula_context_over_chunk_noise():
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from api.main import app

    noisy_chunk = {
        "id": 9,
        "doc_id": "deck.pptx",
        "content": "GE PROPRIETARY noisy extracted slide wall 15,000 random text",
        "source_refs": [{"filename": "deck.pptx", "slide": 3}],
        "metadata": {"slide": 3},
    }
    formula = {
        "id": 1,
        "doc_id": "deck.pptx",
        "chunk_id": 9,
        "formula_text": "BTE = brake power / fuel energy input",
        "formula_latex": None,
        "context_before": "Brake thermal efficiency.",
        "context_after": "Brake power is divided by fuel energy input.",
        "page_or_slide": 3,
        "topic": None,
        "source_title": "Thermal Deck",
    }

    with patch("agent.retriever_hybrid.search", return_value=[noisy_chunk]), \
         patch("agent.reranker.rerank", return_value=[noisy_chunk]), \
         patch("api.main._search_formulas", return_value=[formula]), \
         patch("agent.kg_retrieval.retrieve_kg_context", return_value={"intent": {}, "paths": [], "graph_evidence": [], "context": ""}), \
         patch("llm.stream_chat") as mock_stream_chat, \
         patch("memory.store.get_conversation", return_value=[]), \
         patch("memory.store.append_turn", return_value=None):
        client = TestClient(app)
        resp = client.post("/ask/stream", data={"question": "thermal efficiency formula", "mode": "formula"})

    assert resp.status_code == 200
    assert "## Formula" in resp.text
    assert "BTE = brake power / fuel energy input" in resp.text
    assert "GE PROPRIETARY noisy" not in resp.text
    mock_stream_chat.assert_not_called()


def test_ask_stream_includes_optional_kg_metadata_and_prompt_context():
    """Diagnostic ask/stream keeps SSE shape while adding optional KG evidence."""
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from api.main import app

    chunk = {
        "id": 1,
        "doc_id": "manual.pdf",
        "content": "Engine misfire may be caused by spark plug fouling.",
        "source_refs": [{"filename": "manual.pdf", "page": 12, "topic": "Verbrennungsmotoren"}],
        "metadata": {
            "topic": "Verbrennungsmotoren",
            "relative_path": "02. Verbrennungsmotoren/manual.pdf",
            "source_title": "Hirth Manual",
        },
    }
    kg_result = {
        "intent": {"intent": "diagnostic_cause", "is_kg_relevant": True, "triggers": ["why"]},
        "paths": [{
            "path": "Misfire --CAUSED_BY--> Spark Plug Fouling --FIXED_BY--> Replace Spark Plug",
            "evidence": "Engine misfire may be caused by spark plug fouling.",
            "confidence": 0.86,
            "doc_id": "manual.pdf",
            "page": 12,
            "source_chunk_id": 4,
        }],
        "graph_evidence": [],
        "context": "Knowledge Graph evidence:\nPath 1:\nMisfire --CAUSED_BY--> Spark Plug Fouling",
    }

    def fake_stream_chat(messages, **kwargs):
        assert "Knowledge Graph evidence" in messages[0]["content"]
        assert "Most likely cause" in messages[0]["content"]
        assert "Recommended actions" in messages[0]["content"]
        return iter(["Check ", "the spark plug."])

    def fake_search(query, k=None, topic=None):
        assert topic == "Verbrennungsmotoren"
        return [chunk]

    with patch("agent.retriever_hybrid.search", side_effect=fake_search), \
         patch("agent.reranker.rerank", return_value=[chunk]), \
         patch("agent.kg_retrieval.retrieve_kg_context", return_value=kg_result), \
         patch("agent.verifier.is_grounded", return_value=True), \
         patch("agent.recommender.related", return_value=[]), \
         patch("api.main._fetch_images_for_citations", return_value=[{"url": "/images/manual_page12_img1.png", "page": 12, "citation_n": 1}]), \
         patch("memory.store.get_conversation", return_value=[]), \
         patch("memory.store.append_turn", return_value=None), \
         patch("llm.stream_chat", side_effect=fake_stream_chat):
        client = TestClient(app)
        resp = client.post(
            "/ask/stream",
            data={
                "question": "Why does the engine misfire?",
                "session_id": "kg-test",
                "expertise": "expert",
                "topic": "Verbrennungsmotoren",
            },
        )

    events = []
    for line in resp.text.strip().split("\n\n"):
        if line.startswith("data: "):
            import json as _json
            events.append(_json.loads(line[6:]))
    done = next(e for e in events if e["type"] == "done")
    assert done["intent"]["intent"] == "diagnostic_cause"
    assert done["kg_paths"][0]["path"].startswith("Misfire --CAUSED_BY")
    assert "citations" in done
    assert done["citations"][0]["source_type"] == "kg"
    assert done["citations"][1]["topic"] == "Verbrennungsmotoren"
    assert done["citations"][1]["relative_path"] == "02. Verbrennungsmotoren/manual.pdf"
    assert done["images"] == []
    assert not any(e.get("type") == "images" for e in events)


def test_ask_stream_emits_conflicts_for_cross_document_values():
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from api.main import app

    chunks = [
        {
            "id": 11,
            "doc_id": "spec.pdf",
            "content": "Maximum cylinder head temperature is 220 °C.",
            "source_refs": [{"filename": "spec.pdf", "page": 4}],
            "metadata": {"source_title": "Spec Sheet", "page": 4},
        },
        {
            "id": 12,
            "doc_id": "manual.pdf",
            "content": "Cylinder head temperature limit is 210 °C.",
            "source_refs": [{"filename": "manual.pdf", "page": 8}],
            "metadata": {"source_title": "Maintenance Manual", "page": 8},
        },
    ]

    def fake_stream_chat(messages, **kwargs):
        assert "CONFLICTING VALUES" in messages[0]["content"]
        return iter(["WARNING: Conflicting values found."])

    with patch("agent.retriever_hybrid.search", return_value=chunks), \
         patch("agent.reranker.rerank", return_value=chunks), \
         patch("agent.kg_retrieval.retrieve_kg_context", return_value={"intent": {"intent": "general_question"}, "paths": [], "graph_evidence": [], "context": ""}), \
         patch("agent.verifier.is_grounded", return_value=True), \
         patch("memory.store.get_conversation", return_value=[]), \
         patch("memory.store.append_turn", return_value=None), \
         patch("llm.stream_chat", side_effect=fake_stream_chat):
        client = TestClient(app)
        resp = client.post("/ask/stream", data={"question": "max cylinder head temperature", "session_id": "conflict-test"})

    events = []
    for line in resp.text.strip().split("\n\n"):
        if line.startswith("data: "):
            import json as _json
            events.append(_json.loads(line[6:]))

    done = next(e for e in events if e["type"] == "done")
    conflict_event = next(e for e in events if e["type"] == "conflicts")
    assert done["conflicts"][0]["severity"] == "conflict"
    assert conflict_event["conflicts"][0]["unit"].lower() == "°c"


def test_ask_stream_replaces_false_refusal_when_sources_exist(caplog):
    import logging
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from api.main import app

    chunk = {
        "id": 304,
        "doc_id": "far33.49.pdf",
        "content": "Endurance test. Each engine must be subjected to an endurance test that includes a total of 150 hours of operation.",
        "source_refs": [{"filename": "FAR33.49.pdf", "page": 1}],
        "metadata": {"source_title": "FAR33.49.pdf"},
    }
    fallback = (
        "### Endurance test requirement\n\n"
        "An endurance test is a required engine run sequence used to demonstrate durability. "
        "The cited regulation states that each engine must complete 150 hours of operation.\n\n"
        "---\n**Sources**\n[1] FAR33.49.pdf - Page 1"
    )

    caplog.set_level(logging.WARNING)
    with patch("agent.retriever_hybrid.search", return_value=[chunk]), \
         patch("agent.reranker.rerank", return_value=[chunk]), \
         patch("agent.kg_retrieval.retrieve_kg_context", return_value={"intent": {"intent": "general_question"}, "paths": [], "graph_evidence": [], "context": ""}), \
         patch("memory.store.get_conversation", return_value=[]), \
         patch("memory.store.append_turn", return_value=None), \
         patch("llm.chat", return_value=fallback) as fallback_chat, \
         patch("llm.stream_chat", return_value=iter(["I don't have enough information in the uploaded documents to answer that question."])):
        client = TestClient(app)
        resp = client.post("/ask/stream", data={"question": "what is endurance test?", "session_id": "refusal-test"})

    assert resp.status_code == 200
    assert "I don't have enough information" not in resp.text
    assert "The uploaded documents do contain relevant information" not in resp.text
    assert "### Endurance test requirement" in resp.text
    assert "150 hours of operation" in resp.text
    assert "FAR33.49.pdf" in resp.text
    assert "_stream_answer: LLM refused despite" in caplog.text
    assert any(
        "The user asked a question and these source excerpts were retrieved." in call.args[0][0]["content"]
        for call in fallback_chat.call_args_list
    )


def test_looks_like_refusal_is_short_answer_only():
    from api.main import _looks_like_refusal

    assert _looks_like_refusal("I don't have enough information in the uploaded documents.")
    long_answer = (
        "### Found information\n\n"
        "The source explains the requirement in detail. "
        "It also says users sometimes cannot find information about this phrase in older manuals. "
        * 10
    )
    assert len(long_answer) > 300
    assert not _looks_like_refusal(long_answer)


def test_fallback_does_not_return_second_llm_refusal():
    from api.main import _extractive_answer_from_citations
    from unittest.mock import patch

    citations = [
        {
            "n": 1,
            "filename": "FAR33.49.pdf",
            "page": 1,
            "snippet": (
                "§ 33.49 Endurance test. (a) General. Each engine must be subjected "
                "to an endurance test that includes a total of 150 hours of operation."
            ),
        }
    ]

    with patch("llm.chat", return_value="I don't have enough information in the uploaded documents."):
        answer = _extractive_answer_from_citations("What is an endurance test?", citations)

    assert "I don't have enough information" not in answer
    assert "The uploaded documents do contain relevant information" not in answer
    assert "### Endurance test" in answer
    assert "150 hours" in answer
    assert "±3%" in answer
    assert "FAR33.49.pdf" in answer


def test_kg_diagnostic_fallback_synthesizes_paths():
    from api.main import _extractive_answer_from_citations, _looks_like_raw_kg_answer

    citations = [
        {
            "n": 1,
            "filename": "Knowledge Graph",
            "source_type": "kg",
            "snippet": (
                "Misfire --CAUSED_BY--> Weak Spark --FIXED_BY--> Check Ignition\n"
                "Evidence: Built-in two-stroke diagnostic ontology "
                "Misfire --CAUSED_BY--> Incorrect Timing --FIXED_BY--> Check Ignition\n"
                "Evidence: Built-in two-stroke diagnostic ontology "
                "Misfire (symptom) --CAUSED_BY--> Weak Spark (cause)\n"
                "Evidence: Built-in two-stroke diagnostic ontology"
            ),
        }
    ]

    answer = _extractive_answer_from_citations(
        "What are the likely causes and fixes for engine misfire?",
        citations,
    )

    assert "### Likely causes and fixes" in answer
    assert "**Weak Spark**" in answer
    assert "**Incorrect Timing**" in answer
    assert "Check Ignition" in answer
    assert answer.count("**Weak Spark**") == 1
    assert "--CAUSED_BY-->" not in answer
    assert _looks_like_raw_kg_answer("Misfire --CAUSED_BY--> Weak Spark")
    assert _looks_like_raw_kg_answer("Hirth 140 --HAS_SPEC--> 38 C")


def test_fallback_rejects_irrelevant_table_snippets():
    from api.main import _extractive_answer_from_citations

    answer = _extractive_answer_from_citations(
        "What vibration issues and checks are documented?",
        [
            {
                "n": 1,
                "filename": "Fuel_Kraftstoffe_Uebersicht_Daten.xlsx",
                "snippet": "[Table: Uebersicht] NATO-Code | Designation | density | Flash point",
            },
            {
                "n": 2,
                "filename": "Farhan_Comprehensive_Master_Tracker.xlsx",
                "snippet": "[Table: Master Tracker] Company | Category | Location | Priority",
            },
        ],
    )

    assert "### Not enough directly relevant evidence" in answer
    assert "Master Tracker" not in answer


def test_fallback_formats_torque_specs_as_table():
    from api.main import _extractive_answer_from_citations

    answer = _extractive_answer_from_citations(
        "What torque specifications are available?",
        [
            {
                "n": 1,
                "filename": "Simulation_Modelling.pdf",
                "page": 4,
                "snippet": "Max. Torque: 350 N_m @ 1600 - 2400 rpm Cylinder compression ratio: 16.5",
            }
        ],
    )

    assert "### Available torque specifications" in answer
    assert "| 350 Nm | 1600 - 2400 rpm |" in answer
    assert "Cylinder compression ratio" not in answer


def test_resolve_references_expands_short_followup():
    from unittest.mock import patch
    from agent.graph import _resolve_references

    turns = [
        {"role": "user", "content": "What is the torque spec for HM 50?"},
        {"role": "assistant", "content": "HM 50 torque spec is listed in the manual."},
    ]

    with patch("llm.chat", return_value="torque spec for HM 60 engine"):
        assert _resolve_references("what about the HM 60?", turns) == "torque spec for HM 60 engine"


def test_conversation_history_endpoint_returns_last_turns():
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from api.main import app

    turns = [{"role": "user", "content": str(i)} for i in range(25)]
    with patch("memory.store.get_conversation", return_value=turns):
        client = TestClient(app)
        resp = client.get("/conversation/session-1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "session-1"
    assert len(data["turns"]) == 20
    assert data["turns"][0]["content"] == "5"


def test_question_history_endpoint_returns_recent_user_questions_newest_first():
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from api.main import app

    turns = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]
    with patch("memory.store.get_conversation", return_value=turns):
        client = TestClient(app)
        resp = client.get("/history/session-1")

    assert resp.status_code == 200
    assert resp.json()["questions"] == ["second", "first"]


def test_export_answer_endpoint_downloads_plain_text():
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    resp = client.post("/export/answer", json={
        "question": "What is endurance test?",
        "answer": "It is a 150 hour test.",
        "citations": [{"n": 1, "filename": "FAR33.49.pdf", "page": 1, "snippet": "150 hours"}],
        "images": [],
    })

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "attachment;" in resp.headers["content-disposition"]
    assert "What is endurance test?" in resp.text
    assert "[1] FAR33.49.pdf - page 1" in resp.text


def test_specs_endpoints_shape():
    from fastapi.testclient import TestClient
    from unittest.mock import MagicMock, patch
    from api.main import app

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_cur.fetchone.return_value = (1,)
    mock_cur.fetchall.side_effect = [
        [("doc.xlsx", "Sheet1", "Max RPM", "Value", "Max RPM::Value", "6500", "rpm", {"filename": "doc.xlsx"})],
        [("doc.xlsx", 1, 1)],
    ]

    with patch("config.get_connection", return_value=mock_conn):
        client = TestClient(app)
        specs_resp = client.get("/specs?q=rpm&limit=10")
        summary_resp = client.get("/specs/summary")

    assert specs_resp.status_code == 200
    assert specs_resp.json()["facts"][0]["value"] == "6500"
    assert specs_resp.json()["total"] == 1
    assert summary_resp.status_code == 200
    assert summary_resp.json()["documents"][0]["doc_id"] == "doc.xlsx"


def test_documents_list_endpoint_shape():
    from datetime import datetime, timezone
    from fastapi.testclient import TestClient
    from unittest.mock import MagicMock, patch
    from api.main import app

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_cur.fetchall.return_value = [
        ("manual.pdf", "Manual.pdf", 1, "en", datetime(2026, 1, 1, tzinfo=timezone.utc), 12, 3, "Engines")
    ]

    with patch("config.get_connection", return_value=mock_conn), \
         patch("ingestion.knowledge_base.ensure_document_images_table", return_value=None), \
         patch("ingestion.knowledge_base.ensure_formulas_table", return_value=None):
        client = TestClient(app)
        resp = client.get("/documents")

    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["filename"] == "Manual.pdf"
    assert data[0]["chunk_count"] == 12
    assert data[0]["image_count"] == 3


def test_delete_document_endpoint_removes_related_rows_and_invalidates_cache():
    from fastapi.testclient import TestClient
    from unittest.mock import MagicMock, patch
    from api.main import app

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_cur.fetchall.return_value = []
    def fake_execute(*args, **kwargs):
        sql = str(args[0])
        if "DELETE FROM document_images" in sql:
            mock_cur.rowcount = 2
        elif "DELETE FROM chunks" in sql:
            mock_cur.rowcount = 7
        elif "DELETE FROM documents" in sql:
            mock_cur.rowcount = 1
        else:
            mock_cur.rowcount = 0

    mock_cur.execute.side_effect = fake_execute

    with patch("config.get_connection", return_value=mock_conn), \
         patch("ingestion.knowledge_base.ensure_document_images_table", return_value=None), \
         patch("ingestion.knowledge_base.ensure_formulas_table", return_value=None), \
         patch("agent.retriever_hybrid.invalidate_bm25_cache") as invalidate:
        client = TestClient(app)
        resp = client.delete("/documents/manual.pdf")

    assert resp.status_code == 200
    data = resp.json()
    assert data["chunks_removed"] == 7
    assert data["images_removed"] == 2
    invalidate.assert_called_once()


def test_document_versioning_increments():
    """_register_document inserts version=MAX+1 and returns the new version number."""
    from unittest.mock import MagicMock, patch

    mock_conn = MagicMock()
    mock_cur  = MagicMock()
    mock_cur.fetchone.return_value = (2,)   # MAX(version) = 2 → next = 3
    mock_conn.cursor.return_value = mock_cur

    with patch("config.get_connection", return_value=mock_conn):
        from ingestion.orchestrator import _register_document
        version = _register_document("hirth_3503_manual", "manual.pdf", "de")

    assert version == 3
    sql = mock_cur.execute.call_args_list[1][0][0]
    assert "INSERT INTO documents" in sql


def test_ingest_result_has_version_field():
    """IngestResult dataclass must expose a version field."""
    from ingestion.orchestrator import IngestResult

    r = IngestResult(filename="a.pdf", chunks=5, facts=2, skipped_duplicates=1, version=3)
    assert r.version == 3


# --- Large-corpus inventory -------------------------------------------------

def test_inventory_classifies_supported_and_metadata_only_extensions():
    """Inventory classification must avoid parsing while flagging support status."""
    from ingestion.inventory import classify_extension

    pdf = classify_extension("manual.pdf")
    assert pdf["category"] == "documents"
    assert pdf["supported"] is True
    assert pdf["parser_name"] == "pdf_or_ocr"

    pptx = classify_extension("2015-10-01_Isolated_combustion_chamber.pptx")
    assert pptx["category"] == "presentations"
    assert pptx["supported"] is True
    assert pptx["parser_name"] == "pptx"

    cad = classify_extension("housing.step")
    assert cad["category"] == "cad_engineering"
    assert cad["supported"] is False
    assert cad["metadata_only"] is True
    assert cad["skipped_reason"] == "metadata_only_cad"


def test_inventory_scan_fake_folder_metadata_only(tmp_path):
    """Scanner records metadata, groups by topic, and does not require parsing."""
    from unittest.mock import patch
    from ingestion.inventory import scan_inventory

    topic = tmp_path / "18. Drehmomente"
    cad_topic = tmp_path / "01. CAD"
    hidden = tmp_path / ".git"
    topic.mkdir()
    cad_topic.mkdir()
    hidden.mkdir()
    (topic / "torque.pdf").write_text("not parsed", encoding="utf-8")
    (topic / "notes.pptx").write_text("not parsed", encoding="utf-8")
    (cad_topic / "housing.step").write_text("not parsed", encoding="utf-8")
    (hidden / "ignored.pdf").write_text("not parsed", encoding="utf-8")

    with patch("ingestion.inventory.store_inventory_items") as store_mock:
        result = scan_inventory(tmp_path, max_files=20, batch_id="test-batch")

    store_mock.assert_called_once()
    assert result["scan"]["scanned_files"] == 3
    assert result["scan"]["unsupported_count"] == 1
    assert result["summary"]["total_files"] == 3
    assert result["summary"]["supported_files"] == 2
    topics = {row["topic"]: row for row in result["summary"]["by_topic"]}
    assert topics["Drehmomente"]["count"] == 2
    assert topics["CAD"]["unsupported"] == 1


def test_inventory_api_endpoints_shape():
    """Inventory API keeps a stable, demo-friendly JSON shape."""
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from api.main import app

    summary = {
        "total_files": 3,
        "total_size_bytes": 1234,
        "by_topic": [{"topic": "Drehmomente", "count": 2, "size_bytes": 1000, "supported": 2, "unsupported": 0}],
        "by_category": [{"category": "documents", "count": 2, "size_bytes": 1000}],
        "by_extension": [{"extension": ".pdf", "count": 1, "size_bytes": 500}],
        "by_status": [{"status": "discovered", "count": 3}],
    }

    with patch("ingestion.inventory.inventory_summary", return_value=summary), \
         patch("ingestion.inventory.list_inventory", return_value=[{"file_name": "torque.pdf"}]):
        client = TestClient(app)
        resp = client.get("/inventory/summary")
        assert resp.status_code == 200
        assert resp.json()["total_files"] == 3

        listing = client.get("/inventory")
        assert listing.status_code == 200
        assert listing.json()["items"][0]["file_name"] == "torque.pdf"


def test_inventory_ingest_selected_endpoint_dry_run_and_start():
    """Batch endpoint supports dry-run and controlled ingestion modes."""
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from api.main import app

    dry = {"dry_run": True, "selected_count": 2, "total_size_bytes": 100, "skipped_count": 1}
    started = {"dry_run": False, "job_id": "job-test", "status": "completed", "completed": 2, "failed": 0}

    with patch("ingestion.inventory.dry_run_selected", return_value=dry) as dry_mock, \
         patch("ingestion.inventory.ingest_selected", return_value=started) as ingest_mock:
        client = TestClient(app)
        resp = client.post(
            "/inventory/ingest-selected",
            data={
                "topic": "Drehmomente",
                "extensions": "pdf,pptx",
                "max_files": "5",
                "max_file_size_mb": "20",
                "skip_existing": "true",
                "dry_run": "true",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["selected_count"] == 2
        dry_mock.assert_called_once()

        resp = client.post(
            "/inventory/ingest-selected",
            data={
                "topic": "Drehmomente",
                "extensions": "pdf",
                "max_files": "2",
                "kg_enabled": "false",
                "dry_run": "false",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "job-test"
        ingest_mock.assert_called_once()


def test_pptx_chunks_include_slide_metadata():
    """PPTX slide headers should flow into chunk metadata/source refs."""
    from ingestion.parsers import pptx_parser
    from ingestion.chunker import chunk

    doc = pptx_parser.parse(FIXTURES / "sample.pptx")
    chunks = chunk(doc)
    slide_chunks = [c for c in chunks if c["metadata"].get("slide")]

    assert slide_chunks
    assert slide_chunks[0]["metadata"]["slide"] == 1
    assert slide_chunks[0]["source_refs"][0]["slide"] == 1


def test_short_pptx_like_text_still_produces_chunk():
    from ingestion.chunker import chunk
    from ingestion.types import ParsedDoc

    doc = ParsedDoc(
        text="--- Slide 1 ---\nTitle: Air Intake\nFilter inspection",
        metadata={"filename": "short.pptx", "type": "pptx", "file_type": ".pptx"},
        source_ref={"filename": "short.pptx"},
    )

    chunks = chunk(doc)
    assert chunks
    assert chunks[0]["metadata"]["slide"] == 1


def test_diagnostic_fast_path_uses_kg_evidence():
    from unittest.mock import patch
    from agent.graph import _fast_answer

    kg = [{"content": "Symptom: misfire -> Cause: fouled plug -> Fix: replace plug"}]
    chunk = {
        "id": 1,
        "doc_id": "manual.pdf",
        "content": "Spark plug fouling can cause misfire.",
        "source_refs": [{"filename": "manual.pdf", "page": 12}],
        "metadata": {},
    }

    def fake_chat(messages, **kwargs):
        assert "Knowledge Graph" in messages[0]["content"]
        return "## Answer\nUse the KG path and manual evidence. [1] [2]"

    with patch("agent.tools.diagnostic_tree", return_value=kg), \
         patch("agent.tools.hybrid_search", return_value=[chunk]), \
         patch("llm.chat", side_effect=fake_chat):
        result = _fast_answer("kg_fast", "Why is the engine misfiring?", "Why is the engine misfiring?", "en", "expert")

    assert result["citations"][0]["source_type"] == "kg"
    assert result["citations"][0]["filename"] == "Knowledge Graph"


def test_admin_stats_endpoint_shape():
    from fastapi.testclient import TestClient
    from unittest.mock import MagicMock, patch
    from api.main import app

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_cur.fetchone.side_effect = [(10,), (2,), (1,), (3,), (2,), (1,), (4,), (5,), (6,), (7,), (8,), (9,)]

    with patch("config.get_connection", return_value=mock_conn), \
         patch("ingestion.knowledge_base.ensure_document_images_table", return_value=None), \
         patch("ingestion.knowledge_base.ensure_formulas_table", return_value=None):
        client = TestClient(app)
        resp = client.get("/admin/stats")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_chunks"] == 10
    assert data["total_documents"] == 2
    assert data["images_indexed"] == 5
    assert data["formulas_indexed"] == 6
    assert data["downvoted_chunks"] == 7
    assert data["trusted_chunks"] == 8
    assert data["correction_chunks"] == 9


def test_admin_doc_debug_endpoint_shape():
    from datetime import datetime, timezone
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from api.main import app

    class FakeCursor:
        def __init__(self):
            self.last_sql = ""

        def execute(self, sql, params=None):
            self.last_sql = sql

        def fetchone(self):
            sql = self.last_sql
            if "FROM documents" in sql:
                return ("manual.pdf", "Manual.pdf", 1, "en", "data/uploads/Manual.pdf", datetime(2026, 1, 1, tzinfo=timezone.utc))
            if "COUNT(*) FROM chunks" in sql:
                return (2,)
            if "COUNT(*) FROM document_images" in sql:
                return (1,)
            if "COUNT(*) FROM formulas" in sql:
                return (0,)
            if "FROM ingestion_job_items" in sql:
                return None
            if "FROM file_inventory" in sql:
                return ("ingested", None, datetime(2026, 1, 1, tzinfo=timezone.utc))
            return None

        def fetchall(self):
            if "LEFT(content, 200)" in self.last_sql:
                return [(10, "Preview text", {"page": 1})]
            return []

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def rollback(self):
            pass

        def close(self):
            pass

    with patch("config.get_connection", return_value=FakeConn()):
        client = TestClient(app)
        resp = client.get("/admin/doc-debug/manual.pdf")

    assert resp.status_code == 200
    data = resp.json()
    assert data["doc_id"] == "manual.pdf"
    assert data["chunk_count"] == 2
    assert data["image_count"] == 1
    assert data["chunks_preview"][0]["content_preview"] == "Preview text"
    assert data["ingestion_job"]["status"] == "ingested"


def test_admin_low_quality_chunks_endpoint_and_suppress():
    from fastapi.testclient import TestClient
    from unittest.mock import MagicMock, patch
    from api.main import app

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_cur.fetchall.return_value = [
        (42, "manual.pdf", "Manual", "7", -0.7, 3, 3, "Outdated content"),
    ]

    with patch("config.get_connection", return_value=mock_conn), \
         patch("agent.retriever_hybrid.invalidate_bm25_cache") as invalidate:
        client = TestClient(app)
        low_resp = client.get("/admin/low-quality-chunks")
        suppress_resp = client.post("/admin/chunks/42/suppress")

    assert low_resp.status_code == 200
    assert low_resp.json()[0]["id"] == 42
    assert suppress_resp.status_code == 200
    assert any("quality_score = -999" in call.args[0] for call in mock_cur.execute.call_args_list)
    invalidate.assert_called_once()

