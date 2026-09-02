# Evaluation design: sample sizes and ablations

Written 2026-09-02. The single place that answers "how much data do we need,
and what are we varying?"

---

## 1. Two different things called "the corpus"

I conflated these earlier and it produced a wrong claim. They are
independent and scale differently.

| | what it is | today | can it grow? |
|---|---|---|---|
| **Haystack** | everything indexed and searchable | ~16 docs | **yes, freely** — no labelling needed |
| **Labelled set** | questions with known-relevant docs | 50 q / 12 docs | only with human effort |

**The haystack can be pushed to thousands without labelling anything.**
Adding unlabelled documents doesn't change what's relevant to an existing
question — it adds *distractors*, which is precisely what makes ranking
metrics discriminate. Recall@k over 16 documents is easy for any retriever;
over 5,000 it is not.

### Grow the haystack without OCR

The bottleneck is not fetching, it's Docling/RapidOCR on CPU — minutes per
PDF, so thousands of PDFs is days of compute.

**Index abstracts instead.** `SemanticScholarFetcher` already returns
abstracts, and `eval/fixtures.py` proves the pattern works: those 12
documents are abstract-only, chunked directly with no PDF and no OCR. A few
thousand abstract-only distractors are a fetch and an embed away — minutes,
not days.

### The one real hazard: unjudged relevant documents

If a fetched distractor genuinely answers a labelled question but isn't in
its `relevant_source_ids`, it scores as a false positive. This is the
classic incomplete-judgments problem in IR.

Two mitigations, and I'd take the first:

1. **Fetch distractors from domains the labelled questions don't touch.**
   The 12 fixtures span robotics, medical imaging, quantum computing,
   federated learning, speech, agriculture. Pull distractors from
   deliberately distant fields.
2. **Accept and document it.** The bias is *conservative* — it penalises
   retrievers that surface genuinely relevant unjudged documents, so any
   measured difference is a lower bound.

### How big, and how to get there

**Compute does not constrain this.** The sweep caches retrieved sources per
(mode, question), so a 360-config sweep issues 150 Pinecone + 150 Postgres
queries *regardless of corpus size* — sweep time is flat. Embedding is ~13s
per 1,000 abstract-only docs on CPU; fetching 8,192 is ~10 bulk pages.
Nothing here is expensive.

**The constraint is unjudged relevance.** A fetched paper that genuinely
answers a labelled question, but isn't in its `relevant_source_ids`, scores
as a false positive. That risk grows with corpus size and grows *fastest*
for papers topically close to the labelled ones — which are also the papers
that make retrieval hard. That tension, not compute, sets the size.

**Target 2,048, reached in stages.** Don't guess the number: grow until the
metrics stop saturating, measuring at each step. Each stage is minutes.

| stage | docs | why stop and look |
|---|---|---|
| now | 16 | |
| 1 | 256 | first point where ranking has real competition |
| 2 | 1,024 | |
| 3 | **2,048** | expected landing spot |
| 4 | 4,096+ | only if scores are still saturated at 2,048 |

Beyond ~2k, more *far-domain* documents add nothing — they were already
trivially rejected — while more *near-domain* documents add unjudged-relevant
risk faster than they add difficulty. **Difficulty comes from composition,
not count.**

### Composition

| slice | share of 2,048 | fetch queries | risk |
|---|---|---|---|
| labelled fixtures | 12 | — | — |
| **far-domain** | ~1,700 | fields no question touches: marine biology, medieval history, polymer chemistry, macroeconomics, geology | none |
| **near-domain** | ~350 | same broad fields as the fixtures — legged locomotion, medical imaging, federated learning | real, so it's capped |

Two guards on the near-domain slice:

- **Exclude papers matching a fixture.** The 12 fixtures are real papers
  carrying synthetic `eval-fixture-*` ids. Fetching the same paper under its
  real Semantic Scholar id creates a duplicate that *is* relevant and *isn't*
  labelled. Filter fetched titles against fixture titles before inserting.
- **Keep the near slice small and named**, so if a result looks odd the
  suspect list is 350 documents rather than 2,000.

### Implementation

`PdfChunker` only ever reads `objects` rows, so a document with an abstract
and no PDF is **never chunked and never searchable**. `eval/seed.py` works
around this for the fixtures by inserting Document + a synthetic Object +
Chunk directly.

Distractors need the same path, so: **`eval/distractors.py`** — fetch by
query, drop anything matching a fixture title, insert
Document + synthetic Object (`status='chunked'`) + one Chunk holding the
abstract, then let the existing embed worker pick them up. No PDF, no OCR,
no changes to the pipeline.

```bash
uv run python -m eval.distractors --far 1700 --near 350
```

Worth doing alongside: download PDFs for a few dozen documents so the UI
demo has real papers to preview at a cited page. Distractors don't need
them; the demo does.

---

## 2. Sample sizes — computed, not assumed

Paired design (both configs answer the same questions), McNemar, α=0.05,
assuming the two configs disagree on ~20% of questions.

### Power to detect a difference in recall

| Δ recall | n=42 *(today)* | n=100 | n=200 | n=400 | n=800 |
|---|---|---|---|---|---|
| 0.02 | **0.04** | 0.06 | 0.08 | 0.13 | 0.21 |
| 0.03 | **0.05** | 0.09 | 0.15 | 0.25 | 0.47 |
| 0.05 | **0.09** | 0.18 | 0.34 | 0.61 | **0.90** |
| 0.10 | **0.28** | 0.64 | **0.91** | 1.00 | 1.00 |

**This invalidates a conclusion already in the repo.** `config.py` says
hybrid beats semantic — 0.833 vs 0.810 recall, Δ=0.023 — and the power to
detect Δ=0.02 at n=42 is **0.04**. That result is indistinguishable from
noise. It may well be true; the experiment cannot say so.

The same applies to `keyword_weight: 0.1`, chosen from a sweep on the same
50 questions and already hedged in its own comment as "a sensible default
rather than a tuned constant". That hedge was correct.

### Abstention is far worse

8 questions. A proportion over 8 items has a 95% CI of **±0.35**.

| n | CI half-width | power to separate 0.5 from 0.8 |
|---|---|---|
| **8** *(today)* | **±0.346** | **0.28** |
| 20 | ±0.219 | 0.59 |
| 50 | ±0.139 | 0.92 |
| 100 | ±0.098 | 1.00 |

The README's *"abstention_precision 0.000 → 0.500"* is 0 of 8 versus 4 of 8.
That is not a measurement.

### Cohen's κ for judge selection

| n labels | CI half-width (κ≈0.7) | power to separate κ=0.60 vs 0.80 |
|---|---|---|
| **81** *(have)* | ±0.100 | **0.79** |
| 150 | ±0.073 | 0.97 |
| 300 | ±0.052 | 1.00 |

**The 81 labels are adequate** for the judge experiment as scoped —
separating a good judge from a mediocre one. They cannot rank two good
judges against each other (κ=0.75 vs 0.80 is inside the interval).

### Targets

| set | today | target | why that number | effort |
|---|---|---|---|---|
| **Haystack** | ~16 | **2,000–5,000** | distractors, so MRR/nDCG discriminate | fetch + embed, no OCR |
| **Retrieval questions** | 42 | **200** | power 0.91 at Δ=0.10, 0.34 at Δ=0.05 | the expensive one |
| **Abstention questions** | 8 | **50** | power 0.92 | **nearly free — see below** |
| **Faithfulness labels** | 81 | keep | adequate for the judge decision | done |

**Abstention questions are the cheapest labels in the project.** A question
about a topic the library doesn't cover needs *no relevance annotation* —
the label is the empty set. Writing 42 more is an hour, and it upgrades the
most discriminating axis from unusable to solid.

Retrieval questions are the costly ones because someone must identify which
document answers each. Three sources, cheapest first: `eval.feedback` (#34,
already built — thumbs from real searches become labels), LLM-generated
questions from fixture text with human verification, and hand-writing.

---

## 3. Ablations

### Tier 1 — Retrieval · $0, no LLM

Merges what are currently two separate scripts, because they measure two
halves of one trade-off.

| axis | values | n |
|---|---|---|
| `mode` | semantic · keyword · hybrid | 3 |
| `top_k` | 1, 3, 5, 10, 20, 50 | 6 |
| `rerank` × `rerank_candidates` | off · on×{10,20,40} | 4 |
| **`min_rerank_score`** | **none, −10, −8, −6, −5** | **5** |
| | | **= 360 configs** |

**Why the threshold has to be in the grid.** `eval.sweep` never varies a
score floor, so retrieval always returns `top_k` and `abstention_precision`
is structurally **0.000** for all 72 current configs. The threshold *is* the
abstention mechanism; measuring it in a separate script means the two halves
never appear on the same row.

**Scoring: treat answer-vs-abstain as binary classification.** The dataset is
already labelled for it:

| | returned something relevant | returned nothing |
|---|---|---|
| retrieval questions | TP | FN |
| abstention questions | FP | TN |

Gives **answer-decision precision / recall / F1** over all questions, from
data `score_question` already produces (`hit_rate` is the TP indicator,
`abstention_precision` the TN indicator).

**Select on the Pareto frontier of nDCG × decision-F1, not on nDCG alone.**
Thresholds cost nDCG and buy abstention, so argmax-nDCG *always* picks "no
threshold" — which is exactly why every ledger row reads
`abstention_precision: 0.0` even for the threshold sweeps.
`eval/ingest.py:_best` needs to record the Pareto row.

Report recall@k, MRR, nDCG, decision-F1. **Demote precision@k to a
footnote**: 36 of 50 questions have exactly one relevant document, so
precision@k is `1/k` by construction — arithmetic, not a retriever property.

### Tier 2 — Indexing · $0 but hours of OCR

One knob at a time against the Tier 1 winner. Never crossed with Tier 1.

| axis | values | cost |
|---|---|---|
| `neighbour_window` | 0 · 1 · 2 | **free** — retrieval-time only, no re-index |
| `chunk max_tokens` | 256 · 512 | full re-index |
| `embedder.model` | bge-small · bge-large | re-index + second Pinecone index |

Do `neighbour_window` first: it's the only one that costs nothing.

### Tier 3 — Generation · ~$0.05 total

| axis | values | question |
|---|---|---|
| pipeline shape | fixed · agentic | **why is agentic faithfulness 0.564 vs 0.829?** |
| generator model | minimax:free · one paid | is the gap the model or the shape? |
| `temperature` | 0 · 0.7 | does sampling drive hallucination? |
| prompt version | orchestrator v1 · v2 | the fix, if the gap is real |

A free generator plus a ~4¢ judge makes this essentially free. **The
faithfulness gap is the only genuinely open finding in the project** — it
deserves the attention the retrieval sweeps have been getting.

### Tier 4 — Judge · ~$0.18

Already designed in `docs/judge-selection-experiment.md`; the 81 labels are
in place and adequate.

---

## 4. What this costs, and what it doesn't

| tier | configs | $ |
|---|---|---|
| 1 Retrieval | 360 | **0** |
| 2 Indexing | 3–6 | **0** + OCR time |
| 3 Generation | ~16 runs | ~0.05 |
| 4 Judge | 4 judges | ~0.18 |
| **total** | | **≈ $0.25** |

**Money is no longer the constraint** — that was the point of the OpenRouter
and judge work. The constraints are now sample size and OCR throughput, and
neither is fixed by spending.

---

## 5. Order

1. **Grow the haystack to a few thousand abstract-only distractors.** Every
   retrieval number is measured on a 16-document haystack until this
   happens.
2. **Write 42 more abstention questions.** An hour, no annotation, and it
   converts the most discriminating axis from unusable to solid.
3. **Tier 1** with the threshold axis merged in.
4. **Tier 4** — settle the judge before trusting any judged number.
5. **Tier 3** — the faithfulness gap.
6. **Grow retrieval questions toward 200** — slowest, most valuable
   long-term. Start the feedback loop now so it accumulates in the
   background.
7. **Tier 2** — only if Tier 1 plateaus.

---

## 6. What to say in the writeup

Two of the repo's stated conclusions are underpowered — hybrid-beats-semantic
(Δ=0.023 at power 0.04) and the abstention numbers (n=8).

**Reporting that is better than quietly re-running until they look
significant.** "I measured it, then measured whether the measurement could
support the claim, and it couldn't" is a stronger demonstration of
evaluation literacy than any single number in the table. The fix — more
questions, bigger haystack, confidence intervals on every reported figure —
is in this document.

Every metric in the final table should carry a CI. A number without one is
what produced the two claims above.
