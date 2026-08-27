---
category:
  - "[[Notes]]"
created_at: 2026-08-27
topics:
  - "[[Technical Portfolio]]"
created_by:
  - "[[AI Generated]]"
---
# Why our embeddings live in Pinecone, not Postgres

*Draft design note.*

Papers Please already needs Postgres for documents, PDFs, and chunk text.
Search also needs nearest-neighbor lookup over chunk embeddings. The
question wasn't whether to use Postgres — it was whether embeddings belong
in it too, or in a separate, purpose-built store.

## 1. What we looked at

**pgvector, in the database we already have.** A Postgres extension that
adds a vector column type and ANN indexes (IVFFlat, HNSW). One database,
one connection pool, one backup story. A chunk's text and its embedding
would live in the same row, so "embed this chunk" and "record that it's
embedded" are one `INSERT`, one transaction, no way for the two to
disagree.

**Self-hosted Qdrant or Weaviate.** Purpose-built vector search, better
ANN performance than pgvector at large scale, still something we operate
ourselves — a container to run, disk to provision, upgrades to track.
Genuinely a middle ground, but it buys index quality at the cost of
another stateful service, without pgvector's one-transaction guarantee or
a managed service's zero-ops story.

**Pinecone, a managed vector DB.** No index to tune, no capacity to
provision, scales its own storage and compute independent of whatever the
Postgres instance is doing. The tradeoff is exactly the inverse of
pgvector: embeddings now live in a system we don't control, reached over
the network, with no shared transaction with the metadata that describes
them.

## 2. What we picked, and why we're comfortable

We picked Pinecone. At this project's current scale the operational cost
of self-hosting anything — pgvector's index tuning, Qdrant's container —
outweighs the benefit, and Pinecone's serverless pricing means we pay
per-vector, not for a standing service that's mostly idle between fetches.

The real cost isn't operational, though — it's architectural: **the
system now has two sources of truth for the same fact** ("this chunk has
an embedding"), one in Pinecone's index and one in Postgres. We accepted
that split deliberately, and built the join and the failure handling
around it rather than pretending it isn't there.

```python
# services/backend/process/embedder.py
def execute(self, ...):
    ...
    for batch in ...:
        vecs = self._embed(texts)
        self._upsert_vectors(...)          # write 1: Pinecone
        self._record_embeddings(chunk_ids, model_id)  # write 2: Postgres
```

## 3. How it actually works

### Postgres tracks intent, Pinecone holds the vectors

`chunk_embeddings(chunk_id, model_id)` in Postgres is the record of *which*
chunks have been embedded with *which* model — never the vector itself.
`pending()` is defined purely against that table:

```python
def pending(self, model_id: int) -> list[tuple[int, str, int | None]]:
    already_embedded = select(ChunkEmbedding.chunk_id).where(ChunkEmbedding.model_id == model_id)
    stmt = select(Chunk.id, Chunk.chunk_text, Chunk.page_num).where(Chunk.id.not_in(already_embedded))
```

Postgres never needs to know what a vector looks like, only whether one
exists. That's also what makes the schema support multiple embedding
models cleanly — `embedding_models` maps a model to its own Pinecone
index name, so re-embedding the same chunks with a second model is just a
second `model_id`, not a schema change.

### The join happens in application code, not a database

Search queries Pinecone for nearest neighbors, gets back chunk IDs and
scores, then does a second round trip to Postgres to attach the actual
text, title, and authors:

```python
# services/backend/search.py
response = index.query(vector=vec, top_k=top_k, include_metadata=True)
chunk_ids = [int(m["id"]) for m in response["matches"]]
...
rows = session.execute(select(Chunk.chunk_text, ...).where(Chunk.id.in_(chunk_ids)))
```

Pinecone's `id` field is deliberately the Postgres `chunk_id` as a string
— the two systems are joined by a shared key convention, not a foreign
key, because nothing enforces that convention except the code that writes
both sides.

## 4. What we gave up

**Ordering between the two writes isn't atomic.** `_upsert_vectors` runs
before `_record_embeddings`; if the process dies in between, the vector
exists in Pinecone with no matching `chunk_embeddings` row. Because
`pending()` is defined against Postgres, that chunk looks unembedded again
and gets re-embedded and re-upserted — Pinecone's `upsert` is keyed by ID,
so the duplicate silently overwrites rather than duplicating. That's
lucky, not designed: if the write order were reversed, we'd have a
Postgres row pointing at a vector that was never written, and search would
return a chunk ID that resolves fine in Postgres but never surfaces from
Pinecone at all, since it's absent from the index. We haven't built
reconciliation for that direction because it hasn't happened yet — it's a
real gap, not a solved one.

**Every search is a network hop before it's a database query.** Local
pgvector would fold the ANN search into the same query as the metadata
join. Here it's two round trips to two different networks, and Pinecone's
latency and availability are now on the critical path for every `/search`
call, outside our control.

**Cost scales per-vector, indefinitely.** Fine at hundreds of papers;
worth revisiting once the corpus is large enough that Pinecone's bill
stops being background noise. The `embedding_models` → `index_name`
indirection means a future move to pgvector (or Qdrant) is a new
`SearchEngine` implementation behind the same interface, not a schema
rewrite — but until this pipeline runs at real volume, whether pgvector
would actually be cheaper *and* fast enough is untested, not assumed.
