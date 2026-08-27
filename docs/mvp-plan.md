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

## Orchestrator design decisions

- **LLM**: Claude Haiku 4.5 by default for routing decisions (cheap, fast,
  good tool-calling; a tick is ~500-1500 input tokens deciding among ~5
  tools, well under $0.01/tick). Architecture stays provider-agnostic —
  see below — so swapping to a self-hosted vLLM model is a config change,
  not a code change.
- **Framework**: LangGraph. Small explicit state graph (poll status → route
  → call one tool → loop), which fits LangGraph's node/edge model better
  than a hand-rolled Anthropic SDK tool loop once we want conditional
  routing and retries.
- **Hotswap**: both Claude and vLLM speak to LangGraph through the same
  `BaseChatModel` interface. vLLM exposes an OpenAI-compatible server, so
  `ChatOpenAI(base_url=...)` against it is a drop-in for `ChatAnthropic`.
  One config flag (`llm.provider: anthropic | vllm`) picks the constructor;
  nothing else in the graph changes.

  ```python
  # services/orchestrator/llm.py
  from langchain_anthropic import ChatAnthropic
  from langchain_openai import ChatOpenAI

  def make_llm(cfg) -> BaseChatModel:
      if cfg.llm.provider == "anthropic":
          return ChatAnthropic(model=cfg.llm.model or "claude-haiku-4-5")
      if cfg.llm.provider == "vllm":
          return ChatOpenAI(
              base_url=cfg.llm.vllm_url, api_key="EMPTY",
              model=cfg.llm.vllm_model,
          )
      raise ValueError(cfg.llm.provider)
  ```

## LangGraph primer

You haven't used LangGraph before, so a quick orientation before subtask 4:

- **Core idea**: instead of a single loop that calls an LLM and executes
  whatever tool it asks for, you describe your agent as a **graph** — nodes
  are steps (call the LLM, run a tool, check a condition), edges say what
  runs next. LangGraph runs the graph until it reaches an end state.
- **State**: a graph has a shared `State` (usually a `TypedDict` or
  Pydantic model) that flows through every node — e.g. `{"messages": [...]}`.
  Each node reads state, returns a partial update, LangGraph merges it in.
- **The prebuilt agent** (`create_react_agent`) is the fast path: give it an
  LLM + a list of `@tool`-decorated functions + a system prompt, and it
  builds the classic "LLM picks a tool → tool runs → result goes back to
  LLM → repeat until no more tool calls" graph for you. This is what
  subtask 4 uses — it's the same shape as a manual Anthropic tool-use loop,
  just expressed as a 2-node graph (`agent` node, `tools` node) instead of
  a `while` loop you hand-write.
- **Why bother over a hand-written loop** for something this small: once
  you want branching (e.g. "if `get_status` shows nothing pending, end
  early instead of calling the LLM again"), retries, or to add a second
  agent later, you add nodes/edges to the existing graph instead of
  rewriting loop logic. For v1 here it's mostly free — `create_react_agent`
  is one function call — and it's the on-ramp to writing a custom graph
  later without changing frameworks.
- **What you'll actually touch**: `@tool` functions (plain Python, shown
  below), a system prompt, and — once the prebuilt agent isn't enough —
  `StateGraph` where you define nodes as functions and wire edges yourself
  (including conditional edges, e.g. route to `end` vs `tools` based on
  `get_status` output). We'll start with `create_react_agent` in subtask 4
  and only drop to a manual `StateGraph` if the routing logic outgrows it.

## Proposed subtasks (one GitHub issue each, opened when started)

### 1. Split `worker.py` into independent stage services

Extract `download` / `ocr-chunk` / `embed` into their own entrypoints so
each is a separately deployable/restartable process, instead of three
method calls in one script.

```
services/backend/stages/
  download.py   # entrypoint: python -m stages.download  (loop: PdfFetcher().execute())
  chunk.py       # entrypoint: python -m stages.chunk      (loop: PdfChunker().execute())
  embed.py       # entrypoint: python -m stages.embed      (loop: PdfEmbedder().execute())
```

Each is the existing `while True: run(); sleep(interval)` shape `worker.py`
already has, just scoped to one stage. `PdfFetcher` / `PdfChunker` /
`PdfEmbedder` classes don't change — this is packaging, not new logic.

`compose.yaml` gets one service per stage instead of one `worker`:

```yaml
download-stage:
  build: { context: ./services/backend }
  command: python -m stages.download
worker-chunk:
  command: python -m stages.chunk
worker-embed:
  command: python -m stages.embed
```

### 2. Per-stage poll config

Replace the single `worker.interval_s` in `config.yaml` with one interval
per stage (`stages.download.interval_s`, `stages.chunk.interval_s`, ...) so
they can be tuned independently once they're separate processes.

### 3. Orchestrator tool set

Thin wrappers around the existing `.execute()` calls, expressed as
LangGraph/LangChain tools:

```python
# services/orchestrator/tools.py
from langchain_core.tools import tool

@tool
def fetch_papers(query: str, max_papers: int = 100) -> str:
    """Fetch new paper metadata from Semantic Scholar for a query."""
    n = SemanticScholarFetcher().fetch(query=query, max_papers=max_papers)
    return f"fetched {n} papers"

@tool
def download_pending(limit: int = 20) -> str:
    """Download PDFs for papers that have a URL but no downloaded object."""
    ...

@tool
def chunk_pending(limit: int = 10) -> str: ...

@tool
def embed_pending(limit: int = 500) -> str: ...

@tool
def get_status() -> dict:
    """Return counts of documents/objects per pipeline stage."""
    # SELECT status, count(*) FROM objects GROUP BY status; pending docs w/o pdf_url; etc.
```

`get_status` is what lets the LLM actually route instead of guessing —
it's the one new piece of DB code this subtask needs (a handful of
`SELECT count(*) ... GROUP BY` queries).

### 4. Orchestrator agent loop (LangGraph)

```python
# services/orchestrator/graph.py
from langgraph.prebuilt import create_react_agent

SYSTEM = """You manage a paper-ingestion pipeline: fetch -> download -> chunk -> embed.
Call get_status first. Then call at most one stage tool to make progress on
whichever stage is most behind. If nothing is pending, say so and stop."""

def build_agent(llm):
    tools = [fetch_papers, download_pending, chunk_pending, embed_pending, get_status]
    return create_react_agent(llm, tools, prompt=SYSTEM)
```

`create_react_agent` (LangGraph prebuilt) is enough for a single-agent,
small-tool-set loop — no need for a hand-built graph unless routing logic
grows past "look at status, pick one tool".

### 5. Wire into compose, retire `worker.py`

New `orchestrator` service running the LangGraph loop on a tick
(`interval_s`, same shape as today's worker sleep loop). `worker.py`
deleted once the orchestrator covers the same ground; `PdfFetcher` /
`PdfChunker` / `PdfEmbedder` keep working as they're what the tools call
into — only the *decision of when to call what* moves to the agent.

### 6. Rewrite README

Once the above lands, document the real architecture (independent stage
services + LangGraph orchestrator) instead of the current stub.

Each subtask gets its own GitHub issue per `CLAUDE.md`, opened only when
work starts on it — small, sequential, committed incrementally.
