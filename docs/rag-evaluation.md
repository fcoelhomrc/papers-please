# Evaluating this RAG system

Written 2026-09-03. Supersedes `docs/eval-design.md`. The reference for how
the evaluation is built, what each number means, and what it cannot tell you.

---

## 1. The thing being measured

A RAG answer can fail in two independent places, and conflating them is the
most common way to waste an evaluation:

| failure | example | measured by |
|---|---|---|
| **retrieval** | the passage that answers the question was never fetched | context precision / recall |
| **generation** | the passage was fetched and the model ignored or contradicted it | faithfulness / response relevancy |

A pipeline can score 0.9 on faithfulness while answering the wrong question
entirely, because faithfulness only asks "is every claim supported by what was
retrieved" — not "was the retrieval any good" and not "is the answer correct".
Every metric below is narrow on purpose; the value comes from reading them
together.

---

## 2. Building the test set with Ragas

Ragas generates questions from your own documents. You do not write them.

### The three stages

**Stage 1 — knowledge graph.** Documents become nodes. An LLM extracts a
summary, themes, and named entities from each node; a node filter drops ones
too thin to question. Embeddings then build two kinds of edge between nodes:
cosine similarity over summaries, and entity overlap.

This is where the money goes. In `ragas` 0.2.15 the short-document branch of
`default_transforms` runs **four LLM calls per node** (`SummaryExtractor`,
`CustomNodeFilter`, `ThemesExtractor`, `NERExtractor`). Embeddings are local
and free.

**Stage 2 — synthesis.** Each synthesizer walks the graph differently, picks a
node or cluster, and writes a question plus a gold answer from it. Roughly two
LLM calls per question.

**Stage 3 — human curation.** Not optional. See §5.

### The critical configuration choice

`default_transforms` branches on input length. If ≥25% of your documents exceed
500 tokens it inserts a `HeadlineSplitter`, which re-splits them into new nodes.
That would be fatal here, because it breaks the correspondence between a Ragas
node and one of *our* chunks — and that correspondence is what makes the free
metrics exact (§4).

So we do two things:

1. **Feed Ragas our own chunks**, not our PDFs. Each 256-token chunk from the
   Docling pipeline becomes one `LCDocument` carrying `chunk_id` in its
   metadata.
2. **Pass `transforms=` explicitly** rather than letting `default_transforms`
   choose. No splitter ever runs, so each node's text stays byte-identical to
   a row in the `chunks` table.

We also seed from a **sample** of chunks (~400) while retrieval searches the
**whole** corpus (4,000–8,000 chunks). Questions come from a small pool;
distractors come from everything. That keeps generation cheap and the retrieval
task hard.

### The three synthesizers

Ragas 0.2.15 defaults to ⅓ each. We pass 50/25/25 explicitly.

**`single_hop_specific`** — one chunk, one fact.

```json
{
  "user_input": "What reranking model does the ColBERTv2 evaluation use for MS MARCO?",
  "reference_contexts": ["...we rerank the top-1000 candidates with a cross-encoder based on MiniLM-L-6..."],
  "reference": "A MiniLM-L-6 cross-encoder, applied to the top-1000 retrieved candidates.",
  "synthesizer_name": "single_hop_specific_query_synthesizer"
}
```

**`multi_hop_specific`** — two or more chunks joined by a shared **entity**.

```json
{
  "user_input": "How does the chunk size in the ablation compare to the main results, and why was it changed?",
  "reference_contexts": [
    "...the main experiments use 512-token passages...",
    "...for Table 4 we reduce passages to 128 tokens, since longer passages dilute the relevance signal..."
  ],
  "reference": "Main results use 512-token passages, the ablation 128; the reduction was made because longer passages dilute the relevance signal."
}
```

**`multi_hop_abstract`** — two or more chunks joined by **theme**, not entity.
Broader and comparative.

```json
{
  "user_input": "What trade-offs do these approaches describe between retrieval latency and answer quality?",
  "reference_contexts": ["...late interaction costs 3x the index size but...", "...single-vector retrieval is faster to serve, at a measurable cost in nDCG@10..."],
  "reference": "Late interaction gains accuracy at roughly 3x the index size; single-vector retrieval serves faster but loses nDCG@10."
}
```

### A silent failure worth guarding

`default_query_distribution` calls `get_node_clusters` on each synthesizer and
**drops any that returns nothing** — with no error. The multi-hop synthesizers
need edges that may not exist:

| synthesizer | requires |
|---|---|
| single-hop specific | nodes carrying `entities` |
| multi-hop specific | `entities_overlap_score` relationships |
| multi-hop abstract | `summary_similarity` relationships (cosine ≥ 0.5) |

If your corpus is topically scattered, no two summaries clear 0.5, every
multi-hop synthesizer silently disappears, and you get an all-single-hop test
set that looks fine. We assert cluster counts before generating and fail loudly.

This is also *why* the corpus is built few-topics-deep (§6): the clusters only
exist because papers within a topic genuinely resemble each other.

---

## 3. Mapping questions back to chunk ids

Ragas hands back `reference_contexts` as **strings**. Its own non-LLM metrics
compare those strings to your retrieved strings with Levenshtein distance at a
0.5 threshold.

Because no splitter ran, every `reference_contexts` string is byte-identical to
a `chunks.chunk_text`. So we look each one up in a `{chunk_text: chunk_id}`
dict and store `reference_chunk_ids` alongside.

That converts retrieval scoring from thresholded string comparison into **exact
set arithmetic on integer ids** — deterministic, free, and immune to the
threshold being wrong. It is the single most useful thing in this design.

Stored per row in `eval/testset/generated.jsonl`:

```
id, question, reference, reference_contexts,
reference_chunk_ids, reference_doc_ids, synthesizer, topic
```

---

## 4. The metrics

### Retrieval

| metric | asks | needs |
|---|---|---|
| `LLMContextPrecisionWithReference` | of the chunks retrieved, how many were useful — **and were they ranked first** | reference |
| `LLMContextRecall` | did retrieval get *everything* needed for the gold answer | reference |
| `NonLLMContextPrecisionWithReference` | same, by string match against the seed chunks | reference_contexts |
| `NonLLMContextRecall` | same | reference_contexts |

Precision is mean average precision @k, so **order matters** — this is the
metric that moves when the reranker works. Recall moves when you change `top_k`
or chunk size. They trade off; reporting one alone is gaming.

### Generation

| metric | asks | note |
|---|---|---|
| `Faithfulness` | is every claim in the answer supported by the retrieved chunks | decomposes the answer into statements, runs NLI on each. The hallucination detector |
| `ResponseRelevancy` | does the answer address the question | generates synthetic questions *from the answer* and compares embeddings. Does **not** check correctness |

`ResponseRelevancy` scores noncommittal answers **0**, so a *correct* abstention
is punished. We report abstention separately rather than letting it depress the
mean.

### Why both LLM and non-LLM versions exist

They rest on different ground truth, and each breaks where the other works.

**Non-LLM** asks: did you retrieve the same chunk the question was seeded from?
Free, deterministic, instant.

**LLM** asks a judge: was this chunk useful in reaching the gold answer? It
never looks at `reference_contexts`, so it can credit a chunk the generator
never saw.

Two situations decide which to trust:

1. **A different chunk answers the question just as well.** The question was
   seeded from chunk #47, but #112 in another paper states the same fact, and
   your retriever returns #112. Non-LLM scores a miss; the LLM gives credit.
   This is the classic incomplete-judgments problem — the seed set is a
   *sample* of relevant chunks, not the complete set, and non-LLM treats it as
   complete. **Read non-LLM as a lower bound.**
2. **You changed the chunking.** `reference_contexts` is frozen at generation
   time. Re-chunk at 128 tokens instead of 256 and every retrieved string fails
   the 0.5 threshold — the score collapses even if retrieval improved.
   **Non-LLM metrics are invalid across any chunking change.** LLM metrics
   survive it.

Hence the two branches:

| branch | metrics | when |
|---|---|---|
| **free** | non-LLM precision/recall + exact chunk-id recall@k, precision@k, MRR, nDCG@k | every change, fixed chunking |
| **paid** | faithfulness, response relevancy, LLM precision/recall | manual trigger, and for any chunking/embedding ablation |

One implementation detail decides whether the free branch works at all: score
against `ChunkResult.text`, **not** `.context`. `neighbour_window` glues
adjacent chunks into `context`, which no longer matches the reference string.

---

## 5. What Ragas does not remove

Ragas moves the human from *authoring* to *reviewing* — roughly a 10×
reduction, not elimination. Three duties remain.

**Curating the questions.** Expect to drop 20–30%. Two failure modes recur:

- *The question names its own answer.* "What MiniLM variant..." — BM25
  retrieves the gold chunk on the entity alone, so every configuration scores
  identically and real differences vanish. This one is insidious because the
  question looks perfectly good.
- *The floating "these".* "What trade-offs do **these approaches** describe?"
  is meaningless without knowing which chunks it came from. No user would ask
  it.

If you edit a question, **edit its `reference` too**, or `context_recall`
grades against a stale gold answer.

**Validating the judge.** Every LLM metric is an unvalidated classifier until
checked against human verdicts. `faithfulness = 0.85` means *the judge said
0.85*. The meta-evaluation literature is consistent that this often does not
transfer: judges land at κ 0.3–0.5 against humans while correlation reads
0.8–0.9 on the same data, a gap of 33–41 points. Without labels you cannot
tell "my pipeline improved" from "the judge drifted".

**The gold answers.** `reference` is LLM-written from the seed chunks. A wrong
reference silently penalises correct answers and nothing surfaces it.

The trap specific to this stack: an LLM writes the questions, an LLM writes the
gold answers, and an LLM grades. Same family for all three and you are measuring
self-consistency and calling it accuracy. §7 is the mitigation.

---

## 6. Corpus design

**Few topics, deep — 5 topics × 20 papers.** Retrieval difficulty comes from
*within-topic* neighbours. A hundred papers spread over twenty unrelated fields
is trivial: lexical overlap alone separates them, every configuration scores
~1.0, and the ablation shows nothing. Hard negatives are papers that share
vocabulary and differ in the detail the question asks about.

Topics chosen to be adjacent enough to interfere: retrieval & RAG · agents and
tool use · LLM evaluation & benchmarking · efficient inference & quantization ·
alignment & RLHF. They share half their vocabulary.

Inside each topic, two things are deliberate:

- **Near-duplicates on purpose** — competing methods for the same problem, or
  v1/v2 of an idea. That is where realistic retrieval confusion lives.
- **Varied length** — short workshop papers alongside long surveys, so
  chunk-count skew exercises the neighbour window and the reranker.

40 candidates are staged per topic and a human keeps 20. Taking the top 20 by
citation would produce a *list*, not a corpus.

**arXiv only.** Semantic Scholar's `openAccessPdf.url` points at ACL Anthology,
MDPI or doi.org for most papers, and doi.org landing pages return 403 to a
scripted fetch. Filtering on `externalIds.ArXiv` and **synthesising** the URL
as `https://arxiv.org/pdf/{id}` gives a verified ~100% hit rate.

### Sizing

The corpus must dwarf the seed set or the eval is trivial. Rule of thumb:
**corpus chunks ≥ 20–50× the question count.** 100 questions → 2,000–5,000
chunks → 50–100 papers at 40–80 chunks each.

---

## 7. Model roles

Three jobs, three different families, so nothing grades its own output:

| role | model | why |
|---|---|---|
| test-set generator | `z-ai/glm-5.3-flash` | AA Intelligence 57, highest in the cheap tier; passes strict structured-output tests |
| pipeline answerer | `qwen/qwen3-30b-a3b-instruct-2507` | budget tier, tool-calling, 262k context |
| judge | `deepseek/deepseek-v4-flash` | AA 52, cheapest output tokens of the serious models, parse-tested clean |

Structured-output reliability matters more than raw intelligence here: ragas
metrics parse JSON out of every response, and `gpt-5-nano`
(`LLMDidNotFinishException`) and `gpt-oss-120b` (`OutputParserException`) both
failed on that alone despite being cheap.

Avoid free tiers for generation — 1,000 requests/day does not cover a knowledge
graph build.

---

## 8. Sample size, and reading the numbers honestly

100 questions is the working default: ~±5pp on a mean at 95% confidence. Below
30, noise dominates. Above 300 you can start comparing configurations that
differ by <5pp.

Paired design (both configs answer the same questions), McNemar, α=0.05:

| Δ recall | n=50 | n=100 | n=200 | n=400 |
|---|---|---|---|---|
| 0.02 | 0.05 | 0.06 | 0.08 | 0.13 |
| 0.05 | 0.10 | 0.18 | 0.34 | 0.61 |
| 0.10 | 0.32 | **0.64** | **0.91** | 1.00 |

At n=100, a 5-point difference is **not** detectable. Treat sub-5-point gaps as
ties unless repeats say otherwise, and put a confidence interval on every
reported number.

### Confidence intervals cost nothing

A recurring confusion is that CIs need repeated judge calls. They do not. Two
different variances:

- **Sampling variance** — "would a different 100 questions give a different
  mean?" Estimated from the spread of the 100 per-question scores you already
  have: `SE = s/√n`. Zero extra calls, in the free and paid branch alike.
  Bootstrapping the same 100 scores is equally free.
- **Judge variance** — "would this judge answer differently on a re-run?" This
  does need repeats, but it is a one-time property of the judge, measured once
  during judge selection at temperature 0, not re-measured every run.

For A-vs-B, use **paired** differences — a CI on the mean per-question
difference is tighter than comparing two independent CIs, and still free.

---

## 9. Limits, stated up front

- **The seed set is a sample of relevant chunks, not all of them.** Non-LLM
  metrics therefore undercount. The bias is conservative — it penalises
  retrievers that surface genuinely relevant unjudged chunks — so a measured
  improvement is a lower bound.
- **Questions are synthetic.** They test what an LLM thought to ask of a
  passage, which is not the distribution of what users ask.
- **One annotator, no inter-annotator agreement.** The usual target is
  LLM-human agreement approaching human-human agreement; with a single reviewer
  there is no human-human baseline.
- **n=100 cannot resolve small differences.** Stated above, repeated here
  because it is the limit most likely to be forgotten when a table looks
  convincing.
