# Closing out Papers Please

Written 2026-09-02. What's left to call this done, in dependency order —
each block is gated by the one above it.

---

## 0. Before anything: the index is stale

Nothing measured or filmed is worth keeping until the corpus is rebuilt.
Chunking changed twice (#28 heading prefixes + boilerplate filter, #29
256-token chunks + neighbour expansion), so **every number in
`eval/ledger.jsonl` describes a retriever that no longer exists**. Any
visual or README claim built on them would be wrong on arrival.

```bash
# in psql
TRUNCATE chunks RESTART IDENTITY CASCADE;
UPDATE objects SET status = 'pending' WHERE status IN ('chunked', 'failed', 'dead');

cd services/backend
uv run python -c "from process.embedder import PdfEmbedder; PdfEmbedder().execute(recreate_index=True)"
```

Costs OCR time, **no tokens**. The workers do the rest on their own.

Also worth doing first: **the corpus is 37 documents, 16 chunked.** That is
thin for an eval story. A couple of `/fetch` runs on distinct topics before
the rebuild makes every downstream number more convincing, and the download
stage now stops retrying dead URLs (#37) so it will actually settle.

---

## 1. OpenRouter

Issue **#18** already carries the full implementation plan and is labelled
`blocked` solely on the API key. Nothing to design — sign up, drop
`OPENROUTER_API_KEY` in `.env`, unblock, implement.

One open sub-decision recorded there: which cheap model becomes the
*pipeline* default. It needs reliable tool-calling for the 4-tool agent —
DeepSeek is the candidate but is unconfirmed and needs a check against
current OpenRouter pricing and tool-call support.

**Keep the judge pinned to `anthropic/claude-haiku-4.5`.** Every judged
score in the ledger was produced by it; changing the judge silently
re-baselines every historical comparison.

## 2. The ablation table

Worth being precise here, because the framing matters for the writeup:

- **`eval.sweep` and `eval.thresholds` are already free.** No LLM, any
  model, any provider — they score retrieval against labels. Run them as
  much as you like; OpenRouter changes nothing about them.
- **What OpenRouter actually buys** is cheap *judged* runs, and therefore a
  model-comparison axis the project has never had: same questions, same
  retrieval, different generator.

So the table has two independent halves:

| axis | cost | source |
|---|---|---|
| retrieval config (mode × k × rerank × thresholds) | free | `eval.sweep`, `eval.thresholds` |
| generator model (haiku vs deepseek vs …) | cheap via OpenRouter | `eval.run --sample 15` per model |
| pipeline shape (fixed vs agentic) | cheap | `--variant fixed` / `--variant agentic` |

Use `--sample 15` while iterating and the full 50 only for rows that go in
the table. Judge calls are disk-cached (#26), so re-running an unchanged
row costs nothing, and every run now records its own token/USD spend — put
that **cost column in the table**. "This configuration scores X for $Y" is
a more interesting claim than the score alone, and it is the thread the
whole eval story hangs on.

**The finding to actually chase**: the agent is *less grounded* than the
fixed baseline it should beat — faithfulness 0.564 vs 0.829, while winning
answer_relevancy 0.740 vs 0.623. It retrieves well and grounds badly. Both
prompts are still `v1` despite the versioning machinery existing precisely
for this. An orchestrator `v2` that forces claim→doc_id binding, scored
against `v1`, is the single most compelling result available — and it is
the one thing here genuinely worth spending tokens on.

## 3. Visuals

Already committed and regenerable: `uv run python -m eval.figures` (8 SVGs,
light+dark) and `uv run python -m eval.summary` (the ledger + PR curves as
one HTML page). **Both must be regenerated after §0** — the current SVGs
describe the old chunker.

New ones the recent work earns:

- **Cost per judged run over time.** The ledger now carries `judge_spend`.
  A chart that goes ~1M → ~90k tokens is the most legible single artefact
  in the project, and it is the one a reader will remember.
- **The ablation table itself**, rendered rather than pasted — the summary
  page already has the table machinery.
- **Before/after retrieval** across the chunking change (§0), if the old
  ledger rows are kept for contrast. Label them honestly as different
  chunkers.

Keep the existing light/dark pair convention; the README uses `<picture>`
with `prefers-color-scheme`.

## 4. README

Already substantially rewritten this session: replay mode, the two-tier
eval loop with cost, chunking/`contextualize`, re-index instructions,
feedback, failed-PDF handling, worker health. What's outstanding:

- Numbers refreshed after §0 — several are quoted inline in prose.
- Architecture diagram: it still shows a single worker; there are three
  independent stage workers now, plus the agent and the eval harness.
- A short "what this demonstrates" section near the top. Right now the
  README opens as documentation for an operator; for a portfolio reader it
  should open with what was measured and what was learned.
- The screenshot in `assets/` predates the citation cards, tool trace and
  worker strip.

## 5. Demo video

### Recording on niri/Wayland

Confirmed against your setup (niri 26.04, Wayland, OBS installed):

**Cursor size — verified with `niri validate`:**

```kdl
cursor {
    xcursor-theme "default"
    xcursor-size 48        // default is 24; 48–64 reads well at 1080p
}
```

Bump it for the recording, put it back after. `niri validate -c <file>`
genuinely rejects bad keys, so test before reloading.

**On the "big cursor" idea generally** — it is the weakest of the options,
and worth saying why. This is a text-heavy demo: search results, citations,
a tool trace, log output. What a viewer needs is to *read* things, and a
large cursor doesn't help with that. Ranked by actual payoff:

1. **Zoom in post, not during capture.** Record 1440p+, then push in on the
   region that matters in the edit. Total control, no live fiddling, and it
   is what polished dev demos almost always do.
2. **Large terminal/browser font from the start.** Cheapest legibility win
   available. Bigger than feels comfortable locally.
3. **Cursor size 48** as above — helps the viewer track *where* you are, not
   *what* you clicked.
4. **`obs-zoom-to-mouse`** (OBS Lua script) — the usual recommendation, but
   verify before relying on it: it needs the global cursor position, which
   Wayland deliberately withholds from clients. It may simply not work
   under niri. Don't build the shot list around it.

**niri-specific**: it's a scrolling tiling compositor, which is
disorienting on video if columns slide around unexpectedly. Record on a
dedicated workspace with one or two columns pinned, and avoid live window
management during the take.

**Keystroke overlay**: nothing installed. `screenkey` is X11-only; on
Wayland look at `wshowkeys` or similar — optional, and only worth it if you
demo keyboard-driven flows.

### What to actually show

The replay mode (#30) is the demo's best friend: `PAPERS_PLEASE_REPLAY=1`
gives a deterministic agent with **no API key and no spend**, so takes are
repeatable and nothing costs money. The five scripted questions cover the
interesting states — a one-search answer, a two-search refinement, an
abstention, a fetch, and a tool failure.

Suggested arc, ~3 minutes:

1. Search → result cards with `meaning`/`exact words` provenance badges,
   preview opening at the cited page.
2. Agent → citation superscripts, source cards, the tool trace expanding.
   Include the abstention question: *"nothing in the library covers this"*
   is a better demo beat than another successful answer.
3. Queue → the worker strip, then stop a worker live and watch it go red.
4. The eval story — the summary page and the cost chart. **This is the
   part that distinguishes the project**; give it real screen time rather
   than treating it as an appendix.

### Editing

No editor installed. `kdenlive` for a real timeline, or `losslesscut` if
it's only trims. Both are one line in your Nix config.

## 6. Publish

- Push (done through #38; keep it that way).
- Repo description + topics — a bare repo with no description reads as
  abandoned regardless of what's inside.
- Pin the summary page or a figure in the README so the first screenful
  shows a result, not a setup guide.
- Consider whether `eval/reports/*.md` should stay committed. They're a
  nice audit trail but they're noisy for a browsing reader.

## 7. Portfolio card

The one-line pitch is now doing real work. The obvious framing is not
"semantic search over papers" — that's a thousand repos. It's:

> An agentic RAG pipeline built alongside its own evaluation harness — with
> the retrieval measured judge-free against labels, the LLM-judged metrics
> costed per run, and the cost of a full evaluation cut from ~1M tokens to
> ~90k.

Lead the card with a figure (the PR curve or the cost chart), not a
screenshot of a search box.

## 8. Blog drafts

Two exist:

- `docs/blog-draft-devlog.md` — the original build, up to the pre-agent
  state. Its "what's next" promises the agent and the eval harness, both of
  which now exist.
- `docs/blog-draft-devlog-2.md` — this session: the million-token
  investigation, the reranker doing a fraction of its job, chunking,
  replay, evidence UI, and the two bugs found by actually running the app.

Both are drafts and both end honestly on open edges. The gap between them
is the agent + eval harness build, which no draft covers — either write a
short bridging piece or fold it into a revised part 1.

**The strongest standalone post is not a devlog.** It's the cost
investigation: *"my 50-question eval cost a million tokens, and three
quarters of it was the judge re-deriving something I already measured for
free."* That has a real finding, a number, and a fix, and it doesn't
require the reader to care about this project. Draft 2 already contains it —
it may be worth extracting rather than shipping as part of a devlog.

---

## Order of work

```
0. rebuild index  ─┬─→ 2. ablation table ──→ 3. visuals ──→ 4. README ──┐
1. OpenRouter    ─┘                                                     │
                                                    5. video ───────────┼→ 6. publish
                                                                        │
                                              7. portfolio card ────────┘
                                              8. blog posts (independent)
```

§8 is the only item not gated by the rebuild — the drafts describe
decisions and bugs, not numbers, so they can be written while OCR runs.
