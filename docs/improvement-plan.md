# Improvement plan — cost, retrieval quality, evidence UI

Reviewed 2026-09-01. Ordered by dependency, not by value. One GitHub issue per
section, opened when work on it starts (per CLAUDE.md).

## Constraint on all of this

**No task in this plan may be verified by spending API tokens.** Everything is
tested against fakes and pre-recorded fixtures. Section 5 builds the replay
harness that makes this true for the agent path; until it exists, agent-facing
work is tested with `tests/fakes.py:FakeToolCallingModel`.

---

## Where the 1M tokens went

Measured by call-shape, not by meter (the meter is section 1's job to add):

| Source | Calls / 50-question run | Share |
|---|---|---|
| `context_precision` (1 call **per retrieved context**) | ~250 | ~35% |
| `answer_relevancy` (`strictness=3` → 3 generations) | ~150 | ~20% |
| `faithfulness` (extract statements, then NLI) | ~100 | ~15% |
| `context_recall` | ~50 | ~7% |
| The pipeline's own answers | ~50–150 | ~23% |

At `claude-haiku-4-5` ($1/$5 per MTok) that is roughly **$2–3 per judged run**.
The problem was never the absolute cost — it was that the number was invisible
and that ~55% of it re-derived, badly, what `eval/sweep.py` already measures for
free.

---

## 1. eval: cut judge cost, add a cost meter

- Drop `context_precision` and `context_recall` from `METRICS`. `eval/sweep.py`
  already scores precision/recall/nDCG/MRR against `relevant_source_ids` with
  zero LLM calls, and does it against labels rather than a judge's opinion.
- `answer_relevancy.strictness = 1` (default 3 → 3 generations per question).
- `--sample N` for a stratified subset (by `category` × `domain`), seeded so a
  given N is reproducible. Iterate on ~15; run all 50 for a ledger entry.
- `LangchainLLMWrapper(judge_llm, cache=DiskCacheBackend(...))` — ragas 0.2.15
  memoizes by prompt hash, so re-running an unchanged dataset costs $0.
- `evaluate(..., token_usage_parser=get_token_usage_for_anthropic)`, then record
  `input_tokens` / `output_tokens` / `usd` in `eval/ledger.jsonl` and the
  markdown report.
- README: document `sweep` / `thresholds` as the default loop and `eval.run` as
  the release step.

Expected: ~1M → ~90k tokens per iteration.

## 2. search: widen the rerank candidate pool

`orchestrator/tools.py:search_chunks` passes `rerank_top_k=top_k`, so the
cross-encoder receives exactly as many candidates as it returns — it reorders 5
items and never rescues a 6th. Same bug in `eval/pipeline.py:FixedPipeline`.

Retrieve `search.rerank_candidates` (new config, default 40), rerank down to
`top_k`. The ledger already implies the headroom: `hybrid top_k=50 rerank=off`
scores recall 0.940 vs `top_k=10 rerank→10` at 0.893. Reranking is local CPU —
this costs zero tokens.

## 3. chunk: heading path, boilerplate filter, richer vector metadata

- Prepend the docling heading path (`chunk.meta.headings`, currently discarded)
  to `chunk_text` before embedding: `"Methods > Ablations\n\n<text>"`.
- Drop boilerplate sections at chunk time (references, acknowledgments, author
  affiliations). These are dead weight in the dense index and actively poison
  keyword search — a references section matches every author surname in the
  corpus.
- Store `doc_id` and `year` in Pinecone metadata alongside `page_num`, enabling
  metadata-filtered search and removing a SQL hydration round-trip.

Requires a re-index. Gate behind a `--recreate-index` run, documented.

## 4. search: small-to-big retrieval

Embed for precision, return for context: after ranking, expand each surviving
chunk with its neighbours (`chunk_index ± n`, same `obj_id`) before handing text
to the LLM. Both columns already exist, so this is one SQL query. Fixes the
no-overlap problem in `HybridChunker` without doubling the index.

## 5. replay fixtures + offline mode  ← unblocks all UI work

A recorded-cassette layer so the agent panel, evidence rendering, and the eval
harness can be exercised with **zero API calls**.

- `PAPERS_PLEASE_REPLAY=1` (or `llm.provider: replay`) swaps `make_llm()` for a
  model that replays recorded turns keyed by a hash of the input messages.
- Fixtures are JSON on disk in `services/backend/eval/replay/`, hand-authored
  (not recorded from a live run, which would cost tokens).
- Covers: a single-search answer, a two-search answer, an abstention, a fetch
  request, and a tool-error turn.
- Same switch makes the frontend usable end-to-end against a backend with no
  `ANTHROPIC_API_KEY` set.

## 6. api: structured evidence + streaming from `/agent/chat`

`ChatResponse` currently returns `reply` plus a list of tool *names*, discarding
the retrieved chunks entirely — while `eval/pipeline.py:AgenticPipeline.answer`
already extracts exactly the evidence the UI needs.

- Lift that extraction into one shared helper so eval and the API cannot drift.
- Add `evidence: [{doc_id, chunk_id, title, page_num, score}]` and a
  `trace: [{tool, args, summary, ms}]` to `ChatResponse`.
- Stream via `agent.astream_events` over SSE so tool calls appear as they
  happen rather than as a post-hoc receipt.

## 7. frontend: search result backlinks

- `#page=${page_num}` on the `PdfPreview` iframe src — the page number is
  already on the result card. Highest felt-value-per-line in the repo.
- A real `/documents/:id` route (metadata, chunk list, search-within-paper), so
  a paper is linkable instead of living only inside a modal.
- Carry `sources: ["dense", "keyword"]` through `rrf_fuse` to a badge — answers
  "why did this match?" from data already computed and thrown away.

## 8. frontend: agent evidence cards + tool trace

- Numbered citation cards under each answer, from section 6's `evidence`.
- Post-process `doc_id` mentions in the reply into clickable superscripts that
  open the PDF at the cited page.
- Collapsible trace strip: tool name, args, result summary, duration.
- A distinct abstention state — `min_rerank_score` lets retrieval honestly
  return nothing, and today that renders identically to a failure.

## 9. feedback capture

Postgres, not Phoenix: Phoenix only sees LangChain spans, so `/search` (which
never touches LangChain) produces nothing to annotate, and its volume is
disposable dev state. Feedback exists to grow `eval/dataset.jsonl`, so it has to
be durable and queryable.

One table (`feedback`: id, kind, query, chunk_id, doc_id, verdict, note,
created_at), a `POST /feedback` endpoint, thumbs on search results and on agent
citations, and an `eval/ingest_feedback.py` that proposes new dataset rows.

## 10. queue page + requeue attempt counter

- `PdfChunker._requeue_failed` re-queues failures whenever nothing is pending,
  with no attempt counter — a permanently malformed PDF retries forever, burning
  docling+OCR CPU in a loop. Add `objects.attempts`, cap it, add a `dead` status.
- The Queue page shows aggregate counts only. List the actual papers and the
  stage each is in, so "what is it doing right now" is answerable.

---

## Considered and deferred

**Contextual retrieval** (LLM-written per-chunk document context before
embedding) — the known large win for chunk quality, but a per-chunk LLM call at
ingest. Skipped by decision: it is a one-time cost rather than per-query, so
revisit if section 3's cheap chunking fixes don't move the sweep numbers.

**Query rewriting / multi-query / HyDE** — one extra LLM call per search to
bridge question phrasing and paper phrasing. Not the current bottleneck: the
agent's weak metric is `faithfulness` (0.564 vs the fixed baseline's 0.829), not
`answer_relevancy` (0.740 vs 0.623). It retrieves well and grounds badly, so
spend the next prompt iteration on claim→doc_id binding, not on finding more
candidates. Revisit if recall on the sweep stalls below ~0.9 after section 3.

**Auth / rate limiting on `/fetch`** — not deployed, so not a real exposure.

## The finding this plan does not yet address

```
agentic: faithfulness 0.564  answer_relevancy 0.740  context_precision 0.574
fixed:   faithfulness 0.829  answer_relevancy 0.623  context_precision 0.653
```

The agentic path is substantially **less grounded** than the fixed baseline —
more on-topic, more hallucinated. Both prompts are still at `v1` despite the
prompt-versioning machinery existing precisely for this. After section 1 makes
iteration cheap, the next thing to spend a judged run on is an orchestrator
`v2` that forces claim→doc_id binding, scored against `v1`.
