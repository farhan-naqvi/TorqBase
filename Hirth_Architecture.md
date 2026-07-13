# TwoStrokeGPT — System Architecture (format-aware + enhanced retrieval)

An AI-powered two-stroke knowledge platform for Hirth Engines. Knowledge comes from **user-uploaded documents in any format** (no external scraping in this phase). A **format router** sends each upload to a dedicated handler, then everything converges into normalize → chunk → enrich → embed → dedup. A **ReAct diagnostic agent** retrieves (hybrid + cross-encoder re-rank), answers with cited, self-checked responses in the user's language (EN/DE), and recommends related questions.

```mermaid
flowchart TD
    U["User uploads knowledge files<br/>.pdf · scanned .pdf · images · .docx · .doc · .xlsx · .csv · .pptx · .url"] --> RT

    %% ===== GRAPH 1 — format-aware ingestion pipeline (LangGraph) =====
    subgraph G1["GRAPH 1: LangGraph Ingestion Pipeline (format-aware)"]
        RT{"Format router + language detect (DE/EN)<br/>format_router.py"}

        H1["Digital PDF to text + layout<br/>pdf_parser.py · PyMuPDF"]
        H2["Scanned PDF / images to OCR de+en<br/>ocr.py · Tesseract"]
        H3["DOCX to text + tables<br/>docx_parser.py"]
        H4["Legacy DOC: convert then parse<br/>doc_convert.py · LibreOffice/antiword"]
        H5["XLSX / CSV table-aware<br/>rows · cols · units · formulas<br/>sheet_parser.py"]
        H6["PPTX to slide text + notes<br/>pptx_parser.py"]
        H7["URL / shortcut: extract link only, no fetch<br/>link_handler.py"]

        N["Normalize to clean text + tables + metadata<br/>corpus_builder.py"]
        D["Chunk prose · keep tables intact<br/>chunker.py"]
        E["Extract tags + entities de/en<br/>domain_enricher.py"]
        F["Extract relationships<br/>graph_builder.py"]
        EMB["Embed multilingual<br/>knowledge_base.py"]
        DEDUP["Dedup at ingest · cosine > 0.98<br/>merge provenance, keep all source refs<br/>dedup.py"]

        RT -->|PDF text| H1
        RT -->|scanned/img| H2
        RT -->|docx| H3
        RT -->|doc legacy| H4
        RT -->|xlsx/csv| H5
        RT -->|pptx| H6
        RT -->|url| H7
        H1 --> N
        H2 --> N
        H3 --> N
        H4 --> N
        H5 --> N
        H6 --> N
        N --> D --> E --> F --> EMB --> DEDUP
    end

    %% ===== stores =====
    H1 -. raw file .-> S1
    H5 -. structured rows .-> S6
    H7 -. link reference .-> S4
    N -. clean text .-> S2
    DEDUP -. vectors .-> S3
    E -. metadata .-> S4
    F -. relationships .-> S5

    S1[("Raw file storage<br/>Supabase object store")]
    S2[("Clean corpus<br/>text + table metadata")]
    S3[("Vector DB<br/>pgvector")]
    S4[("Metadata DB<br/>PostgreSQL · tags · links")]
    S5[("Knowledge graph<br/>Engine·Part·Symptom·Cause·Fix")]
    S6[("Structured facts<br/>specs · fuel data · calc tables")]

    %% ===== GRAPH 2 — ReAct diagnostic agent =====
    Q["User asks question or starts diagnosis<br/>EN / DE"] --> G2
    subgraph G2["GRAPH 2: ReAct Diagnostic Agent"]
        R1["Reason: query · model · symptom · intent"]
        R2["Choose tool"]
        R3["Act: run tool"]
        R4["Observe evidence"]
        R5["Draft answer + citations + confidence"]
        R6["Grounding verifier: claims vs sources"]
        R1 --> R2 --> R3 --> R4 --> R1
        R1 --> R5 --> R6
        R6 -->|unsupported or incomplete| R1
        R6 -->|grounded| OUT["Final cited answer (user's language)"]
    end

    TOOLS["Tool layer<br/>hybrid_search · graph_lookup · spec_lookup · conflict_check · unit_convert · diagnostic_tree · source_viewer"]
    RR["Re-rank: top-20 to top-5<br/>cross-encoder · reranker.py"]
    S2 --> TOOLS
    S3 --> TOOLS
    S4 --> TOOLS
    S5 --> TOOLS
    S6 --> TOOLS
    TOOLS -->|top-20 candidates| RR
    RR -->|top-5| R3

    OUT --> REC["Recommend: related questions<br/>+ graph neighbours · recommender.py"]
    S5 -. neighbours .-> REC
    REC --> UI["Web UI<br/>Upload · Chat · Search · Graph · Diagnose · Gaps · Feedback<br/>(token streaming · live upload progress)"]
    OUT --> UI
    UI --> FB["Feedback<br/>thumbs · correction · expert note"]
    FB --> M[("Memory + review store<br/>conversation · profile · approved corrections")]
    M -. reweights retrieval .-> RR
    R6 --> GAP["Gap detector<br/>missing specs · procedure · weak evidence"]
    GAP --> GL[("Knowledge gaps log")]
    GL --> UI

    LLM["LLM<br/>Hosted API or local Ollama (offline)"] -. used by .-> E
    LLM -. used by .-> R1
    LLM -. used by .-> R6

    %% ===== STRETCH / phase 2 — figure & diagram intelligence (added at the end) =====
    FIG["Figure handler — STRETCH / phase 2<br/>extract figures · vision-caption · OCR callouts<br/>figure_handler.py"]
    S7[("Figure store<br/>extracted diagram images")]
    H1 -. figures .-> FIG
    H2 -. figures .-> FIG
    FIG -. caption text .-> D
    FIG -. callouts to parts .-> F
    FIG -. images .-> S7
    S7 --> TOOLS
    LLM -. vision-caption .-> FIG

    classDef stretch stroke:#b45309,fill:#fef3e2,stroke-dasharray:5 4;
    class FIG,S7 stretch;

    %% Roadmap (phase 2): document versioning — same filename → delete old vectors by doc_id → re-ingest
```

## Figure & diagram intelligence (stretch / phase 2)

Handles the visual content text/OCR alone can't — exploded parts diagrams, schematics, and charts in manuals, the thesis, and the simulation report. Added as an end-of-pipeline branch so it's clearly optional, not MVP.

1. **Extract figures** from PDFs (PyMuPDF) and store the image (`Figure store`).
2. **Vision-caption** each figure with a multimodal LLM → a searchable text description ("exploded view, crankcase assembly, callouts 1–18") that flows into the normal chunk → embed path.
3. **OCR callouts → parts** so a number like "14" links to its part name and into the Engine→Part graph.
4. **`source_viewer` returns the actual image** beside the cited answer.

Gets ~80% of the wow ("ask about a diagram, get an answer + the picture") without solving hard spatial reasoning. True "point at the exact bolt" QA is out of scope.

## Retrieval & UX enhancements (what's new in this version)

- **Re-ranking layer (`reranker.py`)** — retrieval returns top-20 candidates; a cross-encoder (e.g. `ms-marco-MiniLM-L6`) re-scores them against the exact query and passes the top-5 to the agent. Big precision gain for numeric/spec queries where "related" isn't good enough. Memory feedback reweights at this stage.
- **Recommendation (`recommender.py`)** — after each answer, suggests related follow-up questions (from retrieved content) and related parts/symptoms (from graph neighbours). This is an explicit challenge requirement, now covered.
- **Dedup at ingest (`dedup.py`)** — cosine > 0.98 means a duplicate (e.g. a spec copied between a manual and a spreadsheet). The duplicate vector is skipped, but **all source references are merged onto the kept chunk** — so provenance and `conflict_check` still see every source.
- **Streaming + progress (UI)** — answers stream token-by-token (after the grounding check passes); uploads show live pipeline progress (Parsing… Chunking… Embedding… N chunks indexed). Demo polish, low effort.
- **Document versioning (roadmap, phase 2)** — re-uploading the same filename prompts replace-or-new-version; replace deletes old vectors by `doc_id` and re-ingests, preventing stale-chunk conflicts. Designed-for, not built for the hackathon.

## How each format is handled (mapped to real sample files)

| Sample file | Type | Handler | Notes |
|---|---|---|---|
| `Diplomarbeit Auslegung und Optimierung.pdf`, `Simulation_Modelling.pdf` | Digital PDF | `pdf_parser.py` (PyMuPDF) | Section-aware chunking + heading/page metadata; German text. |
| `FAR33.49.pdf` | Regulatory PDF (EN) | `pdf_parser.py` | Clause-level chunks; cite by section number (e.g. "FAR 33.49"). |
| *(older scanned manuals)* | Scanned PDF / image | `ocr.py` (Tesseract de+en) | OCR to recover text; flag low-confidence pages for review. |
| `Berechnung Schallgeschwindigkeit im Auspuff.xlsx`, `Fuel_Kraftstoffe_Übersicht_Daten.xlsx` | Spreadsheet (calc/data) | `sheet_parser.py` | **Table-aware**: preserve rows/cols/units (and formulas). Rows written to the **Structured-facts store** so exact values are retrievable + citable. |
| `Mögliche Werkzeugradien.doc` | Legacy Word (.doc) | `doc_convert.py` | `.doc` ≠ `.docx`; convert via LibreOffice/antiword first, then parse. |
| *(modern Word)* | .docx | `docx_parser.py` | Extract text + tables. |
| *(decks)* | .pptx | `pptx_parser.py` | Slide text + speaker notes. |
| `HYDAC International.url` | Windows shortcut | `link_handler.py` | Not a document — extract the target URL, store as a **reference link** in metadata. No content fetch under current scope; flag for optional phase-2 ingestion. |

## Key design decisions driven by this data

- **Numbers must be grounded, never guessed.** This corpus is quantitative (speed of sound, fuel specs, tool radii, FAR test limits). `spec_lookup` reads the **Structured-facts store** and the grounding verifier rejects any numeric claim not tied to a source cell/clause.
- **Spreadsheets are first-class, not stretch.** Two of seven sample files are calculators/data tables — table-aware parsing is in the MVP.
- **German-first multilingual.** Most content is German; multilingual embeddings let an EN or DE query retrieve from either, with answers in the user's language and citations to the original.
- **Handle awkward formats gracefully.** Legacy `.doc` conversion and a clear `.url` policy are built in.

## How to read the graph

**Graph 1 — Ingestion:** format router → per-format handler → normalize → chunk (tables kept intact) → tag/entity extraction → relationship extraction → embed → dedup. Writes to six stores including a dedicated structured-facts store.

**Graph 2 — ReAct agent:** reason → choose tool → act → observe → loop; retrieval is hybrid then cross-encoder re-ranked (top-20 → top-5); draft → grounding verifier loops back if unsupported or emits a cited answer; the recommender adds related questions; weak evidence feeds the gap detector.

**Feedback + memory:** UI feedback → memory/review store → reweights re-ranking. Improves the system without retraining the base model.

**Deployment:** the LLM can be a hosted API or a local Ollama model for offline, data-sovereign operation.
