# Papers, Please

<p align="center">
  <img src="https://raw.githubusercontent.com/fcoelhomrc/papers-please/master/assets/logo.jpg" width="100" />
</p>

<p align="center">
    Fetch scientific papers, index them, and search or ask questions over your library.
</p>

## What it does

- **Fetch** papers from Semantic Scholar by search query (deduped against what you already have — re-fetching a query you've already run adds nothing, doesn't re-download).
- **Download → OCR/chunk → embed**, automatically, as three independent stage workers polling the DB on their own schedule. No queue service — Postgres row status (`pending` / `chunked` / `failed`) is the coordination mechanism.
- **Search** two ways: semantic (vector similarity via Pinecone, optional cross-encoder rerank) and keyword (Postgres full-text) — same result shape, pick whichever fits the question.
- **Chat with an agent** that does two things: decide what to fetch from a natural-language request, or answer questions using papers already in the library (with citations back to `doc_id`/page). It remembers the conversation (per browser tab). It does *not* decide when to download/chunk/embed — that's deterministic, not an LLM's job.
- **Queue dashboard** — live counts of what's pending at each pipeline stage.

## Architecture

```
Semantic Scholar ──fetch──▶ documents (Postgres)
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              worker-download  worker-chunk  worker-embed   (independent,
                    │             │             │            poll-driven,
                    ▼             ▼             ▼            no queue)
                 PDF files    chunks (PG)   Pinecone vectors
                                              + chunk_embeddings (PG)

FastAPI (api.py) ── /fetch /search /search/keyword /status /documents ──▶ Postgres, Pinecone
                 └── /agent/chat ──▶ orchestrator agent (LangGraph + Claude)
                                       tools: fetch_papers, get_status, search_chunks, get_document

React frontend ── docs-style sidebar (Search / Fetch / Documents / Queue) + floating chat panel
```

Each stage worker is a separate process/container (`stages/download.py`, `stages/chunk.py`, `stages/embed.py`), independently restartable and independently tunable (`config.yaml`'s `stages.*.interval_s`/`limit`). They don't know about each other or about the orchestrator agent — the agent is additive (fetch decisions + retrieval), not a replacement for the pipeline.

See [`docs/mvp-plan.md`](docs/mvp-plan.md) for the subtask-by-subtask build history and the reasoning behind these choices (including a couple of corrections made along the way — worth reading if "why does the agent not control downloading" seems like an odd design choice).

## Quickstart

Requires `podman`/`podman-compose` (or Docker Compose — `compose.yaml` is standard).

```bash
cp .env.example .env   # fill in the values below
podman-compose up -d --build
```

Then open **http://localhost:8080**.

### `.env`

| Var | Needed for |
|---|---|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Postgres (documents, chunks, pipeline state) |
| `PINECONE_API_KEY` | Vector storage/search (free tier is enough — one small index) |
| `ANTHROPIC_API_KEY` | The chat agent (`/agent/chat`). Everything else works without it. |
| `HF_TOKEN` | Optional — higher Hugging Face rate limits for model downloads |

## Project layout

```
services/
  backend/        FastAPI app, stage workers, orchestrator agent
    stages/       download.py / chunk.py / embed.py - independent poll loops
    orchestrator/ tools.py, graph.py, llm.py - the LangGraph agent
    ingest/       Semantic Scholar fetch, PDF download
    process/      OCR/chunking (Docling), embedding
    search.py     semantic + keyword search
    status.py     pipeline status query (shared by /status and the agent's get_status tool)
  db/             schema.sql
  frontend/       React + Vite + Tailwind + Radix UI, no other framework
```

## Testing

```bash
cd services/backend
uv run pytest              # fast, mocked - default
uv run pytest -m integration   # real Postgres (disposable podman container) + real Pinecone (isolated test namespace)
```

Every subtask that touches the DB/Pinecone/search has both: mocked unit tests for wiring, and integration tests proving the actual query/index behavior against real infra (mocks agreeing with themselves isn't proof the SQL is right).
