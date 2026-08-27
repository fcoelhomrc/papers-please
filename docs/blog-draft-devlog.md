---
category:
  - "[[Notes]]"
created_at: 2026-08-27
topics:
  - "[[Technical Portfolio]]"
created_by:
  - "[[AI Generated]]"
---
# Papers Please: a semantic search pipeline for scientific papers

*Draft dev log — where the project stands today.*

Papers Please fetches scientific papers, downloads their PDFs, extracts
and chunks the text, embeds the chunks, and serves ranked semantic search
over the result. Four services, three datastores, one pipeline. This is a
walk through what got built, the decisions behind the non-obvious parts,
and where the system's edges are today.

## The shape of the system

```
Semantic Scholar API → Postgres (metadata) → disk (PDFs) → Docling/RapidOCR
  → chunks (Postgres) → embeddings (Pinecone) → FastAPI /search → React UI
```

Four services: a FastAPI backend (`/fetch`, `/search`, `/documents`), a
worker that runs the ingestion pipeline, Postgres, and a React frontend.
Everything is Dockerized and comes up with one `docker-compose up`.

## Fetch: Semantic Scholar's bulk search endpoint

Paper metadata (title, abstract, authors, venue, year, PDF URL) comes from
Semantic Scholar's `/graph/v1/paper/search/bulk` endpoint — cursor-paginated,
so `SemanticScholarFetcher` walks pages with a `token` until either the
result set or the requested `max_papers` runs out, rate-limiting itself
between pages. Rows are upserted with `ON CONFLICT DO NOTHING` on
`source_id`, so re-running the same query is idempotent: it tops up new
papers, doesn't duplicate old ones.

The API only returns a `pdf_url` when Semantic Scholar has located an
open-access copy — a real fraction of fetched papers never get one, and
the pipeline treats "no PDF URL" as a terminal, not-a-failure state: those
documents just never enter the download stage.

## Storage: three places to keep three different kinds of state

- **Postgres** — documents, PDF-object status, chunk text, and the
  join table mapping a chunk to which embedding models have indexed it.
- **Local disk** — the PDFs themselves, one file per object, atomically
  renamed in from a `.tmp` directory so a killed download never leaves a
  half-written file at its final path.
- **Pinecone** — the embedding vectors, one index per embedding model.

Keeping vectors in a separate managed store instead of a Postgres
extension (`pgvector`) trades transactional consistency for zero
index-tuning ops: embedding a chunk is genuinely two independent writes —
upsert the vector, then record in Postgres that it happened — reconciled
today only by the fact that Pinecone's `upsert` is idempotent on ID, so a
crash between the two writes causes a harmless re-embed rather than
silent data loss. That's closer to lucky than designed, and it's the kind
of seam that matters more once ingestion runs unattended at real volume.

## OCR and chunking: Docling, tied to the embedding model's tokenizer

Docling drives OCR (via RapidOCR, CPU by default) and chunking
(`HybridChunker`) in one pass over the PDF. The chunker is configured
against the *embedding model's own tokenizer* and `max_tokens`, not a
fixed character count — so a chunk boundary is always a boundary the
downstream encoder will actually respect, and switching embedding models
means re-deriving chunk boundaries rather than reusing stale ones.

Failures here are common — scanned garbage, encrypted PDFs, layouts
Docling can't parse — and are handled the same way every other stage
handles failure: mark the object `failed`, keep the batch moving, requeue
for another attempt once the pending queue empties out.

## Embedding and reranking: two models, one optional second pass

Embedding uses `sentence-transformers` (`bge-small` or `bge-large`,
configurable), with `embedding_models` tracking which Pinecone index each
model's vectors live in — enough indirection to run two embedding models
side by side without a schema change, though nothing in the pipeline does
that automatically today; it's a manual re-embed if you switch.

Search is a two-stage retrieve → optionally rerank: Pinecone's ANN search
returns the top-k by cosine similarity, and a cross-encoder
(`ms-marco-MiniLM-L-6-v2`) can re-score those candidates against the raw
query text for a second, more expensive pass. Reranking is opt-in per
request (`rerank=true`) rather than always-on, because the cross-encoder
runs locally and adds real latency for a quality gain that isn't free.

## The pipeline itself: a status column, not a queue

There's no message broker or workflow engine — every stage (`PdfFetcher`,
`PdfChunker`, `PdfEmbedder`) reads rows in whatever status it acts on
(`pending`, `chunked`, `failed`), does its work, and writes the next
status, with `ON CONFLICT DO NOTHING` making every write idempotent. One
worker process runs all three stages in sequence, on a fixed interval:

```python
def run():
    PdfFetcher(...).execute()
    PdfChunker().execute()
    PdfEmbedder().execute()
```

That's small enough to hold in your head, and it's honest about where it
stops scaling: everything shares one process and one machine, so a slow
OCR batch delays embedding even when embedding has nothing to do with it,
and the online API and the batch worker both hit the same database with
no isolation between them.

## What's actually working today

Fetch → download → OCR/chunk → embed → search, end to end, with
reranking, through a real UI. Corrupt PDFs get quarantined and retried.
Re-running any stage is safe. The frontend can trigger a fetch, browse
documents, and search with or without reranking.

## Where the edges are

- **One worker, no independent scaling.** All three ingestion stages
  share a process; there's no way to give embedding more throughput
  without also giving OCR more, or to run them on different hardware.
- **No evaluation.** There's no way today to say whether reranking
  actually improves results, or by how much — search quality is currently
  a matter of eyeballing it.
- **Fixed pipeline, no agent.** The pipeline is a hardcoded sequence, not
  a system that decides what to do next based on state. "Explore how far
  a tool-using agent scales" is the stated goal of the project, and
  nothing here does that yet.
- **Vector/metadata split has an unhandled failure direction.** Covered
  above — a crash between the Pinecone write and the Postgres write, in
  the other order, would leave a chunk Postgres thinks is embedded but
  Pinecone doesn't have.

## What's next

The plan (`docs/mvp-plan.md`) is to close the two gaps that actually
matter for the project's stated goal: split the worker into independently
deployable per-stage services, and put a LangGraph agent in front of them
that decides what to run next from live pipeline state instead of a
hardcoded sequence — plus a Ragas eval harness so the eventual agentic
retrieval path can be compared against today's fixed retrieve+rerank on
real numbers, not vibes. That's the next dev log.
