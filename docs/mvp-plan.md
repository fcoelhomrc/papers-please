# MVP plan: agentic pipeline

## Where things stand today

The system already does real work, just not as described in the pitch:

- `services/backend` is one Python package/image. `api.py` serves search +
  fetch-trigger endpoints. `worker.py` is a `while True: sleep` loop that
  calls three stages **sequentially in one process**: download PDFs → OCR +
  chunk (Docling/RapidOCR) → embed + upsert to Pinecone.
- Postgres already tracks per-object state (`pending` / `chunked` / `failed`
  in `objects.status`), so the stages are logically decoupled even though
  they run in the same loop today.
- Fetch (Semantic Scholar), OCR/chunk, embed, and rerank (cross-encoder) are
  all implemented and working. Reranking is wired into `/search` already.
- Frontend (React) hits the API for fetch/search/documents.

## Gap vs. the target description

1. **"Independent services"** — fetch/download/OCR/embed are methods called
   in sequence inside `worker.py`, not separately deployable/scalable
   services. There's no queue, just a shared DB polled by one loop.
2. **"Agentic pipeline" / "tool-using agent"** — there is no agent. It's a
   fixed cron-like pipeline with a hardcoded step order. Nothing decides
   what to do — a while-loop does.
3. Everything else (Semantic Scholar fetch, OCR, chunking, embedding,
   Pinecone, reranking, Postgres) is already in place and doesn't need to be
   rebuilt, just re-packaged around the two gaps above.

## MVP direction

Don't rewrite what works. Two moves get us from "cron pipeline" to
"agentic pipeline":

1. **Split `worker.py` into independent services** — one process per stage
   (fetch, download, ocr-chunk, embed), each polling Postgres for its own
   pending work and exiting/sleeping on its own schedule. This is mostly an
   extraction: the stage logic in `ingest/fetcher.py`, `process/chunker.py`,
   `process/embedder.py` barely changes, only how they're invoked
   (separate entrypoints/containers instead of one script).

2. **Add a thin orchestrator agent** in front of the stages. Instead of a
   fixed step order, an LLM-driven controller with tool calls
   (`fetch_papers`, `download_pending`, `chunk_pending`, `embed_pending`,
   `check_status`) decides what to run next based on current DB state. This
   is the actual "agentic" and "tool-using" part worth exploring — start
   simple (single-agent, small tool set, one call per loop tick) rather than
   a multi-agent system.

## Proposed subtasks (one GitHub issue each)

1. Extract `download`, `ocr-chunk`, `embed` stages from `worker.py` into
   separate entrypoints; update `compose.yaml` with one service per stage.
2. Give each stage its own poll interval/config instead of one shared
   `worker.interval_s`.
3. Define the orchestrator's tool set (thin wrappers around existing
   `PdfFetcher` / `PdfChunker` / `PdfEmbedder` `.execute()` calls +
   a `get_status()` tool reading table counts).
4. Build the orchestrator agent loop (LLM + tool calling, e.g. via the
   Anthropic SDK) that replaces the fixed step order with a decided one.
5. Wire orchestrator into compose as its own service; retire `worker.py`.
6. Update README with the real architecture once the above lands.

Each subtask becomes its own issue per the workflow in `CLAUDE.md` — small,
sequential, committed incrementally.
