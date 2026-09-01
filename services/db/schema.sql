CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    source_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT[],
    venue TEXT,
    year INT,
    pdf_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE objects (
    id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'chunked', 'failed')),
    downloaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    obj_id INT NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT,
    page_num INT,
    UNIQUE (obj_id, chunk_index)
);

CREATE TABLE embedding_models (
    id SERIAL PRIMARY KEY,
    hf_name TEXT NOT NULL UNIQUE,
    dims INT NOT NULL,
    index_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE chunk_embeddings (
    chunk_id INT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model_id INT NOT NULL REFERENCES embedding_models(id) ON DELETE CASCADE,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (chunk_id, model_id)
);

-- Relevance judgements from whoever is using the app. The eval set is
-- hand-authored (eval/fixtures.py), which is the slowest possible way to
-- grow labels while every search is a labelling opportunity going to waste.
--
-- No FK to chunks: a judgement stays true about a query/document pair after
-- a re-index renumbers or removes the chunk it was made against, and losing
-- labels to ON DELETE CASCADE every time the chunker changes would defeat
-- the point of collecting them.
CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('search', 'citation')),
    query TEXT NOT NULL,
    doc_id INT,
    chunk_id INT,
    verdict TEXT NOT NULL CHECK (verdict IN ('up', 'down')),
    note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_feedback_created ON feedback(created_at DESC);

CREATE INDEX idx_documents_has_pdf ON documents(pdf_url) WHERE pdf_url IS NOT NULL;
CREATE INDEX idx_objects_pending ON objects(status) WHERE status = 'pending';
CREATE INDEX idx_chunk_embeddings_model ON chunk_embeddings(model_id);

-- Keyword search (plain Postgres full-text, not a separate search service -
-- expression index means no extra column/trigger to keep in sync).
CREATE INDEX idx_chunks_text_fts ON chunks USING GIN (to_tsvector('english', chunk_text));
