CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    source_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT[],
    venue TEXT,
    year INT,
    pdf_url TEXT,
    citation_count INT,
    -- Which collection this paper belongs to. 'eval' is the curated corpus
    -- the retrieval ablations measure against; anything fetched ad-hoc
    -- through the UI stays 'main'. Without the split, one paper fetched
    -- mid-experiment changes what the questions are competing against and
    -- every number before it becomes incomparable.
    --
    -- 'candidate' is the staging state: papers fetched for review but not
    -- yet promoted into the corpus. They deliberately never download - see
    -- PdfFetcher.pending().
    corpus TEXT NOT NULL DEFAULT 'main'
        CHECK (corpus IN ('main', 'candidate', 'eval')),
    -- Which of the eval topics it was fetched for. Null outside the eval
    -- corpus. Used to stratify question generation and to slice results,
    -- never exposed to the retriever.
    topic TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE objects (
    id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    -- 'downloading' is the row's state before a file exists: it is written
    -- when a download is first attempted, so a failed download has
    -- somewhere to be recorded. Without it the absence of a row meant both
    -- "not tried yet" and "will never work", and a dead URL was retried
    -- forever with nothing to show for it.
    --
    -- 'failed' means "try again", 'dead' means "stop trying". Collapsing
    -- them loses the ability to tell a transient failure from a PDF that
    -- will never parse - and both retry loops need that distinction to know
    -- when to stop.
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('downloading', 'pending', 'chunked', 'failed', 'dead')),
    -- Chunking attempts so far. Without this the requeue loop flips every
    -- failure back to pending whenever the queue empties, so one malformed
    -- PDF is re-OCR'd forever - and keeps the queue non-empty, which
    -- re-fires the condition.
    attempts INT NOT NULL DEFAULT 0,
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
-- Every retrieval path joins chunks -> objects -> documents and filters on
-- corpus, so this sits on the hot path of the ablation sweeps.
CREATE INDEX idx_documents_corpus ON documents(corpus);
CREATE INDEX idx_objects_pending ON objects(status) WHERE status = 'pending';
CREATE INDEX idx_chunk_embeddings_model ON chunk_embeddings(model_id);

-- Keyword search (plain Postgres full-text, not a separate search service -
-- expression index means no extra column/trigger to keep in sync).
CREATE INDEX idx_chunks_text_fts ON chunks USING GIN (to_tsvector('english', chunk_text));
