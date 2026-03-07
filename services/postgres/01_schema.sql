CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    abstract TEXT,
    authors TEXT,
    venue TEXT,
    year INT,
    publication_date DATE,
    citation_count INT,
    influential_citation_count INT,
    s2_paper_id TEXT UNIQUE,
    s2_url TEXT,
    pdf_url TEXT
);

CREATE TABLE objects (
    id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL,
    path TEXT NOT NULL,
    chunk_status TEXT NOT NULL DEFAULT 'pending',
    FOREIGN KEY (doc_id)
        REFERENCES documents(id)
        ON DELETE CASCADE
);

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL,
    chunk_number INT,
    chunk_text TEXT,
    FOREIGN KEY (doc_id)
        REFERENCES documents(id)
        ON DELETE CASCADE
);

CREATE INDEX idx_documents_s2_paper_id ON documents(s2_paper_id)
CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX idx_objects_doc_id ON objects(doc_id);
