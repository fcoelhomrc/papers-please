---
category:
  - "[[Notes]]"
created_at: 2026-09-01
topics:
  - "[[Technical Portfolio]]"
created_by:
  - "[[AI Generated]]"
---
# Papers Please: making the eval cheap and the answers checkable

*Draft dev log — the round of work that started with a scary number.*

The last dev log ended with three gaps: no evaluation, a fixed pipeline
with no agent, and one worker that couldn't scale its stages
independently. Two of those got closed in the meantime — there's a
LangGraph orchestrator now, and a Ragas eval harness with a committed
ledger of every run.

Then the eval harness produced a number that stopped the project: a
50-question run over a dozen seeded papers was burning **a million
tokens**. This log is about chasing that number down, and about what
chasing it turned up — because most of what was wrong wasn't the eval at
all.

## Where a million tokens actually went

The instinct is to blame the agent: it loops, it re-sends the whole
conversation each turn, cost is quadratic in tool calls. That's real,
and it's about a quarter of the bill. The other three quarters were the
*judge*.

Ragas metrics are not one call each. Counting call-shape rather than
guessing:

| Source | Calls per 50-question run | Share |
|---|---|---|
| `context_precision` — one call **per retrieved context** | ~250 | ~35% |
| `answer_relevancy` — `strictness=3`, so three generations | ~150 | ~20% |
| `faithfulness` — extract statements, then NLI over them | ~100 | ~15% |
| `context_recall` | ~50 | ~7% |
| The pipeline's own answers | ~50–150 | ~23% |

`context_precision` is the one worth staring at: its cost scales with
`k`. Retrieve ten chunks instead of five and the metric silently doubles.

The second thing that came out of the arithmetic: at Haiku 4.5's
$1/$5 per MTok, a million input tokens and a quarter-million output is
**about two or three dollars**. The number was never actually
frightening. What was wrong is that it was *invisible* — a score with no
cost beside it is exactly how a 50-question run grows into a
million-token one without anyone noticing.

So the fix was two-part: measure it, then stop paying for the parts that
weren't buying anything.

**Dropped `context_precision` and `context_recall` entirely.** The
project already had a judge-free retrieval sweep scoring
precision/recall/nDCG/MRR against hand-labelled `relevant_source_ids`,
at zero token cost, against ground truth rather than a judge's *opinion*
of ground truth. Paying an LLM to re-derive a weaker version of a
measurement that already exists is the definition of waste. What's left
is the pair a judge is genuinely required for, because both are
properties of generated prose that no label can capture: is the answer
supported by its context, and does it answer the question.

**`answer_relevancy.strictness = 1`.** Averaging three reverse-question
generations buys stability in the third decimal place, well below the
noise floor of a 50-question set.

**A stratified `--sample N`.** Iterating on a judged run shouldn't cost a
full run. The sampling is round-robin across `category × domain` strata
rather than proportional allotment, because with 50 questions over ~8
strata, proportional rounding drives the two-row `edge_case` stratum to
zero — and those are precisely the rows that catch abstention
regressions. A subset that loses them measures the easy half of the
problem and reports it as the whole. It's seeded, so `--sample 15` means
the same fifteen questions next week.

**An on-disk judge cache.** Ragas ships `DiskCacheBackend`, keyed by
prompt hash. Re-running an unchanged dataset — after a report tweak,
after a crash once the answers were already generated — now costs
nothing.

**A cost meter.** `evaluate(..., token_usage_parser=...)` plus a price
table, and the spend lands in the report, the ledger, and a `cost`
column in the summary page. Retrieval sweeps render as `free` rather
than blank, because a blank reads as "unknown" when the answer is a
definite zero — and that zero is the whole argument for running them.

Net: roughly a million tokens per iteration down to about ninety
thousand.

## The reranker was reordering, not choosing

Chasing the cost turned up something better. The agent's search tool did
this:

```python
engine.search(query, top_k=top_k, rerank=rerank, rerank_top_k=top_k)
```

Candidate pool and output size are the same number. The cross-encoder
was handed five candidates and asked to return five: it could reorder
them, but it could never promote a sixth. A reranker exists to retrieve
wide and cut narrow, and this one had been quietly doing a fraction of
the job it was loaded for.

The ledger already had the evidence lying in it: `hybrid top_k=50
rerank=off` recalls **0.940**, against `top_k=10 rerank→10` at
**0.893**. The relevant chunk is usually *in* a wide pool and merely
ranked too low to survive a narrow one.

Now retrieval fetches a pool (default 40) and the cross-encoder cuts to
`top_k`. It's opt-in per call rather than the new default, which matters
more than it sounds: `top_k` means "retrieve exactly this many" to the
sweep and to the `/search` endpoint, both of which are *measuring
retrieval itself*. Silently retrieving 40 when they asked for 5 would
corrupt the numbers the whole plan is being steered by.

The cross-encoder runs locally on CPU. This one cost nothing at all.

## What actually gets embedded

Docling hands back body text with the section headings stripped out into
metadata — and the heading path was being thrown away. Two papers'
Methods sections describing the same technique became near-identical
vectors with nothing to tell them apart, and a query naming a section
("ablation results") had no term to match against.

Chunks now store their breadcrumb:

```
Methods > Ablation Studies

We ablate the encoder and observe ...
```

The satisfying part: this is free against the token budget.
`HybridChunker` sizes chunks by counting its own *contextualized* form,
headings included — so storing bare text was leaving the window
under-filled, not saving room in it.

**Boilerplate now gets dropped at chunk time** — references,
acknowledgments, funding, competing interests. This isn't tidiness. A
references section is a packed list of author surnames and title
fragments, so it matches the keyword retriever for practically any
author or topic query, and scores high `ts_rank` doing it because the
terms are dense. It's a whole class of confident, useless hits removed
at the source.

Two bugs in that matcher are worth recording, because both were caught
by tests rather than by review:

1. The first version stripped leading enumeration ("6.", "VII -") with a
   roman-numeral character class. `C` is a roman numeral, so
   `"Competing Interests"` normalised to `"ompeting interests"` and
   sailed straight through the filter. The enumeration match now
   requires trailing whitespace, which is what distinguishes
   `"VII - Bibliography"` from a word that merely begins with `C`.
2. Prefix matching killed `"Reference implementation"` — a real Methods
   subsection. It's now an exact match on the whole normalised heading.

Appendices are deliberately *kept*: they routinely hold ablations,
proofs and extra results that questions are actually about. And only the
deepest heading decides, so `"Appendix > Ablation results"` stays while
`"Discussion > Acknowledgments"` goes.

## Small chunks to embed, wide context to read

`HybridChunker` emits no overlap, so a sentence spanning a boundary is
split and neither half reads as an answer. And one chunk size can't be
right for two different jobs: embedding wants small and focused,
generation wants surrounding prose.

So chunks dropped from 512 tokens to 256 — a smaller chunk is a sharper
vector, because averaging a passage that covers two topics lands the
embedding between both and near neither — and the context is recovered
*after* ranking, by gluing each surviving hit to its neighbours.

Three details that took some thought:

- **After reranking, never before.** Expanding first would hand the
  cross-encoder a blur of three chunks to score, and the score would
  stop saying which one matched.
- **`text` stays the chunk that matched; `context` is the window.**
  Widening `text` in place would make the score look like it applied to
  all three. The model reads `context`; the UI shows `text`.
- **Neighbours are found within the same `obj_id`, not the same
  document.** `chunk_index` restarts per PDF, so matching on index alone
  would splice one paper's prose into another's.

Net context per search is roughly flat — five 256-token chunks with
neighbours against the old five 512-token ones — so this buys precision
rather than spending tokens. Worth saying plainly, since it looks like a
3× context increase at first glance.

## Running the whole thing with no API key

Every remaining piece of work was UI over agent responses, and none of
it should need an API key or spend a token to develop against. Before
this there was no way to make the agent panel do *anything* without a
live Anthropic key, a populated Postgres and a Pinecone index all up at
once.

Replay mode swaps **both ends** of the loop — the model that decides and
the tools that fetch — while the LangGraph graph runs for real. That
split is the whole design. Faking only the model would send its scripted
tool calls straight at real Pinecone. Faking the entire agent would
exercise none of the wiring the UI depends on. This way the tool-calling
loop, the checkpointer, the message sequence and everything reading it
downstream all behave exactly as in production: a bug in how a tool
result becomes a citation is catchable here, and only a bug in retrieval
quality is not.

```bash
PAPERS_PLEASE_REPLAY=1 docker compose up
```

Six hand-authored fixtures — a one-search answer, a two-search
refinement, an abstention, a fetch, a tool failure, and a fallback so an
unanticipated question explains itself instead of dead-ending. Written
by hand rather than recorded from a live run: recording would spend the
tokens this exists to avoid, and a hand-written fixture can cover a tool
failure that's awkward to provoke on demand.

The shipped fixtures are tested like code — every scripted tool call has
a result to return, every fixture ends on a spoken answer rather than a
dangling tool call, and no fixture is shadowed by another's match
pattern.

## Answers you can check

The agent was told to cite its sources, and obediently produced
`[doc 3, p4]` — plain text a reader could not click, could not resolve
to a paper, and could not check without going and searching for the
passage themselves. Meanwhile the API kept only `call["name"]` from each
tool call and discarded the retrieved chunks entirely, one function call
before building the response. The evidence was there and was being
thrown away.

Now `/agent/chat` returns `evidence` and `trace` alongside the reply,
extracted by a module the eval harness shares — so a reported score
describes the same retrieval the user is looking at. The panel renders
citations as numbered superscripts that scroll to source cards, and the
cards open the PDF at the cited page.

The decisions that took the longest were the small ones:

- **Citations are numbered per document, not per card.** A paper usually
  contributes two or three passages; numbering by card position made the
  sequence skip (1, 2, then 4) and pointed `[doc 3]` at only the first
  of doc 3's passages. A unit test caught it.
- **A citation to a document that was never retrieved is left exactly as
  the model wrote it.** Silently dropping it would hide the model citing
  something it did not find — which is the single most useful signal in
  the entire panel.
- **The citation regex matches what a model actually writes**, not just
  what the prompt asked for: `[doc 3]`, `[doc 3, p. 4]`, `[Doc 3, page
  4]`, `[doc. 3, pp. 4-5]`. A citation that fails to match renders as
  raw brackets mid-sentence, which is worse than not having tried.
- **Abstention is now distinct from failure.** The rerank score floor
  exists so retrieval can *decline* rather than always hand back its top
  k — and that rendered identically to a search that errored. One means
  "the library doesn't cover this", the other means "try again".

And the panel streams. `POST /agent/chat/stream` emits an event per
graph step, so a search announces what it's looking for while it runs
instead of arriving as a post-hoc receipt.

## Two bugs found by actually using it

Both of these came from launching the app rather than from reading it,
which is its own argument.

**Observability was sitting in the request path.** Driving the real
server, every request took twenty-plus seconds and a two-search agent
turn never visibly finished. Not the agent: `phoenix.otel.register`
defaults to a `SimpleSpanProcessor`, which exports each span *inline on
the thread that produced it*. With no collector listening that's about
six seconds of gRPC connect-and-retry per span, and an agent turn
produces a lot of spans. Its own log warns about the default. Switching
to a batch processor — and skipping registration entirely when no
endpoint is configured — took the same six questions from twenty-second
timeouts to 5–48ms.

**The chunker had an infinite retry loop.** `_requeue_failed` flipped
every failed object back to `pending` whenever the pending queue
emptied, so a PDF that will never parse was re-OCR'd forever. The nasty
part is the feedback: requeueing *keeps the queue non-empty*, which is
the condition that fires the requeue. One malformed file could hold the
chunker at 100% CPU indefinitely, on the most expensive step in the
pipeline.

There's an attempt counter now, incremented at failure time rather than
at requeue time — a crash between the two would otherwise lose the
count, and an uncounted attempt is an infinite retry with extra steps.
Past the cap the object becomes `dead`, which is a separate status from
`failed` on purpose: `failed` means "try again", `dead` means "stop
trying", and only the second is a state the loop can safely leave alone.

## A feedback loop, finally

The eval set was being grown by hand while every search anyone ran was a
labelling opportunity going to waste. A thumbs-up is *exactly* what
`relevant_source_ids` encodes: for this question, this paper is
relevant.

Results and citations now carry thumbs, stored in Postgres — not
annotated onto Phoenix spans, because Phoenix only sees LangChain calls,
so `/search` produces no span to annotate, and its volume is disposable
dev state that labels have to outlive.

`eval/feedback.py` turns them into **proposed** dataset rows rather than
appending them, for two reasons that aren't squeamishness. A dataset row
also needs `ground_truth`, which a thumb cannot supply — so it comes out
blank for a human, because an invented ground truth would poison every
faithfulness score computed against it afterwards. And feedback is
unvetted input: silently appending would let a stray click move the
numbers this project steers by, with no diff to notice it in.

A thumbs-*down* is stored but is not turned into an inverse label.
"This result is bad" says nothing about which paper would have been
good, and `relevant_source_ids` has no way to record a negative.

## What's actually working today

An eval loop with two tiers, and the cheap one is the default: judge-free
retrieval sweeps for every retrieval change, a judged run reserved for
questions about generated prose — and both now report what they cost. An
agent whose answers carry openable, checkable sources and a visible tool
trace. A pipeline that gives up on files that will never parse, and a
Queue page that says what it's doing rather than only how much is left.
And the whole thing runs end to end on recorded fixtures with no API key,
which means the UI can be developed without spending anything.

## Where the edges are

- **The index is stale by construction.** Chunk text and chunk size both
  changed, so every existing vector predates the current chunker. The
  sweep numbers in the ledger describe 512-token chunks without heading
  prefixes and aren't comparable until the corpus is rebuilt. That
  rebuild costs OCR time, not tokens — it just hasn't been run.
- **None of this has been scored against a live judge yet.** The cost
  work was done specifically to stop spending casually, so the first
  real `--sample 15` run is still ahead. Every claim above about
  *quality* is a reasoned expectation; only the cost claims are measured.
- **The agent is still less grounded than the baseline it's supposed to
  beat.** The ledger has said so for a while:

  ```
  agentic: faithfulness 0.564   answer_relevancy 0.740
  fixed:   faithfulness 0.829   answer_relevancy 0.623
  ```

  It retrieves well and grounds badly — more on-topic, more
  hallucinated. Both prompts are still at `v1`, despite prompt
  versioning existing precisely so a candidate can be scored against an
  incumbent.
- **Two schema changes need manual DDL.** There's still no migration
  tooling, so new columns land in `schema.sql` for fresh databases and
  in the README as `ALTER TABLE` for existing ones. That's fine at this
  size and won't be for long.
- **The vector/metadata split still has the failure direction the last
  log described.** Nothing in this round touched it.

## What's next

The obvious next spend is the one thing here that genuinely needs
tokens: an orchestrator `v2` that forces claim→document binding, scored
against `v1` on a stratified sample. The faithfulness gap is the largest
open quality problem in the system, the machinery to measure a fix has
existed for weeks, and it's now cheap enough to iterate on — which was
the entire point of starting with the scary number.

Before that, a corpus rebuild, so the retrieval numbers describe the
retriever that actually exists.
