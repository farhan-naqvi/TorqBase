-- TwoStrokeGPT schema (PostgreSQL + pgvector)
-- NOTE: embedding dimension must match EMBEDDING_DIM in .env (default 384).

CREATE EXTENSION IF NOT EXISTS vector;

-- uploaded documents (one row per file/version)
CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,                 -- stable id across versions (e.g. filename slug)
    version     INT  NOT NULL DEFAULT 1,
    filename    TEXT NOT NULL,
    mime        TEXT,
    lang        TEXT,                          -- 'de' | 'en' | ...
    storage_uri TEXT,                          -- where the raw file lives
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, version)
);

-- embedded text chunks (prose + table descriptions + figure captions)
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    content     TEXT NOT NULL,
    lang        TEXT,
    embedding   vector(384),                   -- <-- keep in sync with EMBEDDING_DIM
    metadata    JSONB DEFAULT '{}'::jsonb,     -- page/section/sheet/type, etc.
    source_refs JSONB DEFAULT '[]'::jsonb      -- MERGED on dedup; list of {doc_id, page, ...}
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS quality_score FLOAT DEFAULT 0.0;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS vote_count INT DEFAULT 0;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS downvote_count INT DEFAULT 0;
CREATE INDEX IF NOT EXISTS chunks_quality_score_idx
    ON chunks (quality_score);

-- extracted figures/diagrams from uploaded documents
CREATE TABLE IF NOT EXISTS document_images (
    id            SERIAL PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    page_or_slide INT,
    image_index   INT,
    file_path     TEXT NOT NULL,
    url           TEXT NOT NULL DEFAULT '',
    caption       TEXT DEFAULT '',
    chunk_id      BIGINT REFERENCES chunks(id),
    created_at    TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS document_images_chunk_idx
    ON document_images (chunk_id);
CREATE INDEX IF NOT EXISTS document_images_doc_location_idx
    ON document_images (doc_id, page_or_slide);

-- lightweight extracted engineering formulas
CREATE TABLE IF NOT EXISTS formulas (
    id             SERIAL PRIMARY KEY,
    doc_id         TEXT NOT NULL,
    chunk_id       BIGINT REFERENCES chunks(id),
    formula_text   TEXT NOT NULL,
    formula_latex  TEXT,
    context_before TEXT,
    context_after  TEXT,
    page_or_slide  INT,
    topic          TEXT,
    source_title   TEXT,
    created_at     TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS formulas_doc_idx
    ON formulas (doc_id);
CREATE INDEX IF NOT EXISTS formulas_topic_idx
    ON formulas (topic);

-- metadata-only large corpus inventory (no parsing/embedding during scan)
CREATE TABLE IF NOT EXISTS file_inventory (
    id                   BIGSERIAL PRIMARY KEY,
    batch_id             TEXT NOT NULL,
    root_path            TEXT NOT NULL,
    relative_path        TEXT NOT NULL,
    absolute_path        TEXT,
    file_name            TEXT NOT NULL,
    extension            TEXT,
    size_bytes           BIGINT NOT NULL DEFAULT 0,
    modified_at          TIMESTAMPTZ,
    topic                TEXT,
    category             TEXT NOT NULL DEFAULT 'unknown',
    parser_name          TEXT,
    supported            BOOLEAN NOT NULL DEFAULT false,
    directly_supported   BOOLEAN NOT NULL DEFAULT false,
    metadata_only        BOOLEAN NOT NULL DEFAULT false,
    needs_converter      BOOLEAN NOT NULL DEFAULT false,
    skipped_reason       TEXT,
    status               TEXT NOT NULL DEFAULT 'discovered',
    error                TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (batch_id, relative_path)
);
CREATE INDEX IF NOT EXISTS file_inventory_batch_idx
    ON file_inventory (batch_id);
CREATE INDEX IF NOT EXISTS file_inventory_topic_idx
    ON file_inventory (topic);
CREATE INDEX IF NOT EXISTS file_inventory_status_idx
    ON file_inventory (status);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id          BIGSERIAL PRIMARY KEY,
    job_id      TEXT UNIQUE NOT NULL,
    batch_id    TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',
    total_items INT NOT NULL DEFAULT 0,
    done_items  INT NOT NULL DEFAULT 0,
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ingestion_job_items (
    id            BIGSERIAL PRIMARY KEY,
    job_id        TEXT NOT NULL REFERENCES ingestion_jobs(job_id) ON DELETE CASCADE,
    inventory_id  BIGINT REFERENCES file_inventory(id),
    status        TEXT NOT NULL DEFAULT 'pending',
    error         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (job_id, inventory_id)
);

-- exact values pulled from spreadsheets/tables (for spec_lookup; never invent numbers)
CREATE TABLE IF NOT EXISTS structured_facts (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    sheet       TEXT,
    row_label   TEXT,
    col_label   TEXT,
    key         TEXT,
    value       TEXT,
    unit        TEXT,
    source_ref  JSONB DEFAULT '{}'::jsonb
);

-- knowledge graph: Engine -> Part -> Symptom -> Cause -> Fix
CREATE TABLE IF NOT EXISTS graph_nodes (
    id     BIGSERIAL PRIMARY KEY,
    type   TEXT NOT NULL,                      -- engine | part | symptom | cause | fix
    name   TEXT NOT NULL,
    props  JSONB DEFAULT '{}'::jsonb,
    UNIQUE (type, name)
);
CREATE TABLE IF NOT EXISTS graph_edges (
    id        BIGSERIAL PRIMARY KEY,
    src_id    BIGINT REFERENCES graph_nodes(id),
    dst_id    BIGINT REFERENCES graph_nodes(id),
    relation  TEXT NOT NULL,
    props     JSONB DEFAULT '{}'::jsonb
);

-- feedback loop (improves retrieval; does NOT retrain the model)
CREATE TABLE IF NOT EXISTS feedback (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT,
    question    TEXT,
    answer      TEXT,
    vote        INT,                           -- +1 / -1
    correction  TEXT,
    expert_note TEXT,
    chunk_ids   BIGINT[],
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- knowledge gaps (questions with weak/no evidence)
CREATE TABLE IF NOT EXISTS gaps (
    id          BIGSERIAL PRIMARY KEY,
    question    TEXT,
    reason      TEXT,                          -- missing spec | missing procedure | weak evidence
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved    BOOLEAN DEFAULT false
);

-- memory: per-session conversation + per-user profile
CREATE TABLE IF NOT EXISTS conversations (
    session_id  TEXT PRIMARY KEY,
    turns       JSONB DEFAULT '[]'::jsonb,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS user_profile (
    user_id     TEXT PRIMARY KEY,
    lang_pref   TEXT,
    expertise   TEXT,                          -- beginner | expert
    engines     TEXT[],
    props       JSONB DEFAULT '{}'::jsonb
);
