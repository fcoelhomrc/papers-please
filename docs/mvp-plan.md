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

### 3. Orchestrator tool set — corrected scope

**Original version of this subtask (and #4 below) wrapped
download/chunk/embed as agent tools too — that was wrong and got reverted.**
Given the DB state, download/chunk/embed each have exactly one correct
action; there's no decision for an LLM to make, so they stay on the
deterministic `stages/*.py` workers (subtask 1) and are never exposed as
agent tools. The only things an LLM decision genuinely adds value to here
are (a) what to search for when fetching, and (b) retrieval (subtask 9).
So the tool set is intentionally small:

```python
# services/backend/orchestrator/tools.py
from langchain_core.tools import tool

@tool
def fetch_papers(query: str, max_papers: int = 100) -> str:
    """Fetch new paper metadata from Semantic Scholar for a query."""
    n = SemanticScholarFetcher().fetch(query=query, max_papers=max_papers)
    return f"fetched {n} papers"

@tool
def get_status() -> dict:
    """Return counts of documents/objects/chunks — context for fetch
    decisions (e.g. don't refetch what we already have), not a router
    for the ingest pipeline."""
    # SELECT status, count(*) FROM objects GROUP BY status; pending docs w/o pdf_url; etc.
```

Lives in `services/backend/orchestrator/` (not a separate top-level
package) — it imports `db`/`config`/`ingest` directly, same as `stages/`.

### 4. Orchestrator agent loop (LangGraph)

```python
# services/backend/orchestrator/graph.py
from langchain.agents import create_agent  # langgraph.prebuilt.create_react_agent, deprecated in v1.0, moved here

SYSTEM = """You decide what papers to fetch into a research library, given a request.
Call get_status if it helps judge whether we already have relevant papers.
Call fetch_papers with a search query capturing what's being asked for.
Downloading/chunking/embedding happen automatically on their own schedule
once papers are fetched - not your job, don't try to trigger them."""

def build_agent(llm):
    tools = [fetch_papers, get_status]
    return create_agent(llm, tools, system_prompt=SYSTEM)
```

`create_agent` is enough for this single-agent, two-tool loop — no need
for a hand-built graph.

### 5. Wire into compose

New `orchestrator` service, invoked on demand (a fetch request) rather
than on a poll-and-loop timer like the stage workers — there's no ongoing
state for it to check, only a decision to make when someone asks for
papers on a topic. `stages/*.py` workers are unaffected and keep running
exactly as they do today; the orchestrator is additive, not a replacement.

### 6. Rewrite README

Once the above lands, document the real architecture (independent stage
services + LangGraph orchestrator) instead of the current stub.

### 7. Eval set + Ragas harness

Without this, "agentic pipeline" is a claim with no evidence. Build a small
eval set and wire up [Ragas](https://docs.ragas.io/) so pipeline changes
(rerank on/off, model swaps, agentic vs. fixed retrieval) are measurable,
not vibes.

```
services/backend/eval/
  dataset.jsonl        # {"question": ..., "ground_truth": ..., "doc_ids": [...]}
  run.py               # loads dataset, runs a pipeline variant, scores with ragas
  results/              # one JSON per (pipeline_variant, timestamp) run
```

```python
# services/backend/eval/run.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

def run_eval(pipeline: Pipeline, dataset_path: str) -> dict:
    rows = [json.loads(l) for l in open(dataset_path)]
    records = []
    for row in rows:
        result = pipeline.answer(row["question"])
        records.append({
            "question": row["question"],
            "answer": result.answer,
            "contexts": result.contexts,
            "ground_truth": row["ground_truth"],
        })
    ds = Dataset.from_list(records)
    scores = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    return scores.to_pandas().to_dict()
```

`Pipeline` is a small common interface both the existing retrieve+rerank
path and the new agentic RAG path (subtask 9) implement, so `run_eval` is
variant-agnostic:

```python
class Pipeline(Protocol):
    def answer(self, question: str) -> AnswerResult: ...  # AnswerResult = {answer, contexts}
```

Eval set starts small (~20-30 hand-written Q/A pairs grounded in a handful
of ingested papers) — enough to catch regressions, not a benchmark claim.

### 8. Prompt versioning

Any prompt sent to an LLM (orchestrator system prompt, agentic RAG's
tool-use system prompt, reranker query-prompt string) gets a version tag so
eval results and issue reports can say *which* prompt produced a score.

```
services/backend/prompts/
  orchestrator/v1.md
  agentic_rag/v1.md
```

```python
# services/backend/prompts/registry.py
def load_prompt(name: str, version: str) -> str:
    return (Path(__file__).parent / name / f"{version}.md").read_text()

PROMPT_VERSIONS = {"orchestrator": "v1", "agentic_rag": "v1"}  # config default, override per eval run
```

Bump `vN` on meaningful prompt edits (new file, old ones kept) rather than
editing in place — so an eval run always records an exact, reproducible
prompt version alongside its score, and old scores stay comparable.

### 9. Optional agentic RAG path (for comparison)

A second answer-producing path, alongside the existing fixed
retrieve→rerank pipeline, implementing the same `Pipeline` interface from
subtask 7. Instead of a fixed top_k retrieve + rerank, an agent iterates:
search, read more if the top chunks look insufficient, optionally refine
the query, then answer.

```python
# services/backend/pipelines/agentic_rag.py
@tool
def search_chunks(query: str, top_k: int = 5) -> list[dict]: ...  # wraps existing SearchEngine

@tool
def get_document(doc_id: int) -> dict: ...  # full doc metadata + more chunks if needed

class AgenticRagPipeline:
    def __init__(self, llm):
        self.agent = create_react_agent(
            llm, [search_chunks, get_document],
            prompt=load_prompt("agentic_rag", PROMPT_VERSIONS["agentic_rag"]),
        )

    def answer(self, question: str) -> AnswerResult: ...
```

Selected via config (`search.pipeline: fixed | agentic`) or as an explicit
`/search` query param — not a replacement for the fixed pipeline, a second
option to run side-by-side in eval (subtask 7) for a results/comparison
section: same eval set, `fixed` vs `agentic` scores, cost/latency deltas.

Each subtask gets its own GitHub issue per `CLAUDE.md`, opened only when
work starts on it — small, sequential, committed incrementally.
