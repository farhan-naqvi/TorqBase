# TwoStrokeGPT — System Architecture

An AI-powered two-stroke knowledge platform for Hirth Engines. Knowledge comes from **user-uploaded documents in any format** (no external scraping in this phase). Two LangGraph graphs: a linear **ingestion pipeline** that turns uploads into a searchable knowledge base, and a **ReAct agent** that answers questions with cited, self-checked responses.

```mermaid
flowchart TD
    UP["User uploads — upload_handler.py<br/>Any format: PDF · scanned (OCR) · DOCX · XLSX · PPTX · images · TXT"]

    %% ===== GRAPH 1 — ingestion pipeline (one linear LangGraph, runs on upload) =====
    subgraph PIPE["GRAPH 1 · Ingestion Pipeline — orchestrator.py (linear LangGraph)"]
        C["collect · upload_handler.py"]
        P["parse + OCR · preprocess.py"]
        CK["chunk · chunker.py"]
        EM["embed · knowledge_base.py"]
        GB["graph build · graph_builder.py<br/>Engine → Part → Symptom → Fix"]
        C --> P --> CK --> EM --> GB
    end

    %% ===== GRAPH 2 — ReAct agent loop (main.py) =====
    subgraph AGENT["GRAPH 2 · ReAct agent — main.py"]
        direction LR
        RE["reason<br/>LLM picks an Action"] -->|Action| AC["act<br/>run a tool"]
        AC -->|Observation| RE
        AC --> GR["ground check · verifier.py"]
        GR -->|not grounded| RE
    end

    STORE[("Supabase storage — uploaded files")]
    CORP[("data/corpus — clean text + metadata")]
    PGV[("pgvector — embeddings")]
    GRAPH[("graph tables — Engine·Part·Symptom·Fix")]
    MEM[("memory — conversation · user profile · feedback")]
    OUT[("results — cited answers + gaps log")]

    RET["retriever_hybrid.py<br/>BM25 + dense fusion (multilingual)"]
    TOOLS["tools.py<br/>vector_search · graph_lookup · spec_lookup · unit_convert"]
    UI["web/app.py<br/>Upload · Chat · Graph · Gaps (EN / DE)"]
    LLM["LLM — hosted API<br/>or Ollama local (offline · data-sovereign)"]

    UP --> C
    C -. writes .-> STORE
    P -. writes .-> CORP
    EM -. writes .-> PGV
    GB -. writes .-> GRAPH
    PGV --> RET
    GRAPH --> RET
    RET --> TOOLS
    AC --> TOOLS
    RE -. reads/writes .-> MEM
    GR --> OUT --> UI
    UI -->|Ask button| AGENT
    UI -->|Upload| PIPE
    UI -->|feedback 👍/👎/correct| MEM
    MEM -. reweights .-> RET
    RE -. calls .-> LLM
    GR -. calls .-> LLM
    EM -. calls .-> LLM
```

## How to read it

**Graph 1 · Ingestion Pipeline** — runs whenever a user uploads a document. Any format is collected, parsed (with OCR for scanned files), chunked, embedded into pgvector, and linked into the Engine→Part→Symptom→Fix knowledge graph.

**Graph 2 · ReAct agent** — answers questions. The LLM reasons, picks a tool (action), observes the result, and loops until ready; a ground-check verifies the draft is supported by sources before replying (corrective RAG), looping back if not.

**Data stores** — Supabase storage (raw files), the cleaned corpus, pgvector (embeddings), graph tables, a memory store (conversation, user profile, feedback), and results (cited answers + gap log).

**Retriever** — hybrid BM25 + dense fusion so exact terms (e.g. part numbers like `3503`) and semantic matches both work; multilingual so a German query matches English manual text.

**Feedback loop** — user feedback (votes / corrections) writes to memory, which reweights retrieval over time. The system improves without retraining the base model.

**Deployment note** — the LLM can be a hosted API or a local Ollama model for offline, data-sovereign operation (relevant for defense / heavy-fuel customers).
