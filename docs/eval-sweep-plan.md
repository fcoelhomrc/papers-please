# Finishing the retrieval evaluation

Written 2026-09-09. **This scope is frozen.** Everything below is what gets
run; the "Not in scope" section at the bottom is what deliberately does not,
and it is a decision rather than an oversight.

Blocks are in dependency order — each is gated by the one above it.

---

## Where this starts from

Four encoders and BM25 are embedded and judged, all at `mode=semantic`
(BM25 at `mode=bm25`), `top_k=10`, **reranking off**:

| | bge-small | bge-large | arctic | qwen | BM25 |
|---|---|---|---|---|---|
| recall | 0.678 | 0.732 | 0.790 | 0.767 | **0.797** |
| nDCG | 0.567 | 0.586 | 0.664 | 0.589 | **0.705** |
| MAP | 0.496 | 0.505 | 0.580 | 0.489 | **0.626** |
| R-precision | 0.440 | 0.425 | 0.488 | 0.380 | **0.585** |
| MRR | 0.593 | 0.593 | 0.704 | 0.605 | **0.773** |
| faithfulness | 0.849 | 0.843 | 0.865 | 0.886 | **0.896** |
| answer_relevancy | 0.848 | 0.821* | **0.875** | 0.831 | 0.866 |

\* scored before the judge-embedder pin (`de0ac35`), so not comparable.

Three facts drive everything below:

1. **BM25 wins 9 of 10 metrics.** Encoder size was never the explanation —
   scaling within bge bought +0.019 nDCG, changing family bought +0.097.
2. **Recall is nearly tied at the top** (BM25 0.797, arctic 0.790, qwen
   0.767) while nDCG spreads 0.589–0.705. The candidates are being found;
   the *ordering* is what differs. Ordering is the reranker's job.
3. **The current reranker makes ranking worse.** On bge-small at top_k=10,
   no-rerank scores nDCG 0.6714 against 0.6536 / 0.6507 / 0.6463 for
   rerank_candidates 10 / 20 / 40. It degrades monotonically with more
   candidates to sort — the signature of a cross-encoder scoring near noise
   on this domain.

---

## 1. Arctic ablations — free

Arctic is the strongest dense encoder and has never been ablated. Three open
questions ride on it, and one sweep answers all three.

```bash
podman-compose run --rm -T worker-embed python -m eval.ablations a b w timed
# config.yaml: embedder.model = arctic-m-v2
```

- **`a`** — does `hybrid_bm25` on arctic beat BM25 alone (0.705)? This is the
  question that decides production `search.mode`. Arctic dense-only is 0.664;
  fusion has 0.041 to make up and two near-parity rankers to do it with.
- **`w`** — `keyword_weight` is stale. It was set to 1.0 when dense scored
  0.567 against BM25's 0.705. At arctic's 0.664 the two sides are close to
  parity, and the weight that suited a weak dense side almost certainly is
  not the right one now.
- **`b`** — rerank on/off with the incumbent, to confirm the bge-small result
  above holds on arctic before block 2 replaces the model.
- **`timed`** — per-stage latency, never measured on arctic.

Cost: **$0**, ~1–1.5h wall clock.

---

## 2. Replace the reranker — free to run, needs code first

The last component never swapped. `ms-marco-MiniLM-L-6-v2` is 22M parameters
trained on 2021 web-search passages, and block 0's fact 3 says it is currently
subtracting quality. This is the same "weakest model in its family" story that
bge-small turned out to be.

**Code, before any measurement:**

- Make `reranker_model` a sweep dimension — today it is a fixed config field,
  so no comparison is expressible.
- **Re-derive `min_rerank_score` per candidate.** The current `-8.0` is
  calibrated to this model's logit scale, and cross-encoders do not share a
  scoring range. Carried over blindly it will either reject everything or
  disable abstention entirely — and that failure reads as "the new reranker is
  broken" rather than "the threshold is wrong."

**Candidates:** incumbent, `BAAI/bge-reranker-v2-m3`,
`mixedbread-ai/mxbai-rerank-base-v2`.

**Measured against arctic and qwen.** Qwen specifically: it has the 2nd-best
recall (0.767) and hit rate (0.900) but the *worst* R-precision (0.380) of all
five arms. It finds the right chunk and buries it, which is the single best
case for a reranker in the whole sweep — and the one most undersold by
rerank-off numbers.

Once a model is chosen, sweep `rerank_candidates` for it. Pool size is only
meaningful relative to a reranker worth feeding.

Cost: **$0** to run, ~1h of implementation + ~1h of runs.

---

## 3. One judged run on the winning configuration — costs money

Every judged number on disk is single-mode with reranking **off**. Production
is `hybrid` with reranking **on**. Nothing measured so far describes the
pipeline that would actually ship.

So: one judged run on the configuration blocks 1 and 2 select — best mode,
best encoder, best reranker, reranking on.

```bash
podman-compose run -d -T -e PAPERS_PLEASE_REPLAY= \
  --name papers-please_judged-final backend python -m eval.run
podman update --restart=no papers-please_judged-final   # see note below
```

This is the payoff run and the one number the writeup is actually about.

Cost: **~$0.19**, ~30m.

---

## 4. Re-run bge-large judged — costs money

Its `answer_relevancy` (0.821) was scored before the judge embedder was pinned
to bge-small, so that one cell used a different yardstick from every other arm.
Only that metric is affected; the LLM-judged metrics and every retrieval number
are unaffected.

**This is the one block to drop if the budget matters.** The alternative is an
asterisk in the results table forever.

Cost: **~$0.18**, ~30m.

---

## Totals

| | time | cost |
|---|---|---|
| 1. arctic ablations | ~1–1.5h | $0 |
| 2. reranker replacement | ~2h | $0 |
| 3. final judged run | ~30m | ~$0.19 |
| 4. bge-large redo | ~30m | ~$0.18 |
| | **~4–5h** | **~$0.37** |

Blocks 1 and 2's runs are unattended. Block 2's code and the model choices
want a person awake.

---

## Operational notes, learned the hard way

- **`podman-compose run -d` inherits the service's `restart: unless-stopped`.**
  A judged run that failed looped 23 times; the embed sweep looped 38 after
  finishing. Always follow a detached run with
  `podman update --restart=no <name>`.
- **`PAPERS_PLEASE_REPLAY=1` lives in `.env`.** It puts the answerer on the
  recorded cassette, which returns `""` for anything unrecorded — 100 empty
  answers, caught by the guard in `eval/run.py` only after all the retrieval
  work is done. Pass `-e PAPERS_PLEASE_REPLAY=` on every judged run.
- **`eval/figures.py:25` uses `parents[3]`**, which resolves on the host but
  not at `/app` in the container, so it breaks test collection there. Run the
  suite on the host: `cd services/backend && uv run pytest tests/ -q`.

---

## Not in scope

Decided against, not forgotten:

- **qwen and bge-large full ablations.** Only qwen's `b` (rerank) is
  interesting, and it is already inside block 2. Neither model is a contender
  for production; their rows in the comparison table are complete enough.
- **Re-measuring query arms (`c`) on arctic.** Answered on bge-small — none of
  multi_query, HyDE or decomposition beat leaving the question alone — and the
  arms are text transforms that do not depend on the encoder.
- **More encoders.** Four is enough to have answered the question, and the
  answer was that the family matters and BM25 still wins.
- **Contextual retrieval.** Already deferred in `improvement-plan.md`, and a
  per-chunk LLM call at ingest is a different order of cost from everything
  here.
- **Chunker structural element types (#46)** and the **review UI (#42).** Real
  work, separate issues, not part of finishing this evaluation.
- **Naturally-phrased questions.** The honest limitation of the whole result:
  the questions were generated *from* the chunks, so they inherit chunk
  vocabulary, and lexical matching is what benefits. The judged branch does not
  escape this — both branches use the same questions, so BM25-favoured
  retrievals propagate into faithfulness and relevancy. Fixing it means writing
  questions by hand, which is a project, not a step. It belongs in the writeup
  as a stated limitation.

---

Per the working agreement, each block gets its own GitHub issue when it is
started — not before.
