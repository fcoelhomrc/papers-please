# Picking an LLM judge, properly

Plan of record. Supersedes the ad-hoc probe recorded in
`docs/judge-model-selection.md`, which chose `deepseek-v4-flash` on evidence
that turns out not to support the choice.

## Why the first attempt didn't decide anything

Three constructed cases, judged once each. Four models scored **identically**
(1.0 / 0.33 / 0.0), which reads as consensus but is really a statement about
the cases: they were too easy to separate anyone. It measured accuracy on
obvious inputs and nothing else — not stability, not sensitivity to subtle
error, not failure rate at volume.

## What the literature says to do instead

Two sources, both directly on point:

- **[Reliability without Validity](https://arxiv.org/html/2606.19544v1)** —
  21 judges, 9 providers, ~541,000 judgments. The largest systematic
  meta-evaluation available.
- **[Evaluating LLM-Evaluators](https://eugeneyan.com/writing/llm-evaluators/)**
  — practitioner-oriented, same conclusions.

The correction that matters: **raw agreement and correlation both flatter a
judge.** The study measured Cohen's κ of **0.3–0.5** where Spearman's ρ read
**0.8–0.9** on the same judgments. They call the gap *kappa deflation* and
found it at **33–41 percentage points** across all 21 judges — an 85%
exact-match score is κ ≈ 0.48, which is *moderate*, not excellent.

My original plan's "require ρ > 0.8" would have passed judges with merely
fair agreement. Eugene Yan is blunt about the same thing: *"I tend to be
skeptical of correlation metrics. They don't account for chance agreement
and thus could be over-optimistic."*

Their **Minimum Viable Validation Protocol**, and what we take from it:

| their rule | here |
|---|---|
| chance-correct — κ or Krippendorff's α, not exact match | **adopted**, κ is the headline |
| swap positions, flag if \|P(A)−0.5\| > 0.10 | **not applicable** — see below |
| ≥3 runs at temperature 0, caching disabled | **adopted** |
| ≥2 benchmarks with different label structures | **adopted** — constructed set + real answers |
| if test-retest > 0.95, verify position bias < 0.10 | **adapted** — statement-order shuffle |

**Position and verbosity bias don't apply.** Those are pairwise phenomena
("is A or B better"). Ragas faithfulness is *pointwise*: decompose an answer
into statements, then judge each one supported / not-supported against the
context. There is no A/B ordering to flip.

That turns out to be a gift. Because the unit of judgment is a **binary
per-statement verdict**, κ is computable directly on verdicts rather than on
averaged scores — a far sharper instrument than comparing final numbers.

The one thing we keep from their bias section is the *consistency–bias
paradox*: high test-retest reliability can mask a judge that is
deterministically wrong. Stability is necessary, not sufficient.

## The measurement

### Experiment 1 — accuracy (Cohen's κ)

Calls ragas' `NLIStatementPrompt` **directly**, bypassing statement
extraction. Extraction is itself an LLM step; leaving it in would mix
extraction variance into a number meant to be about the verdict. It is also
much cheaper, since ~5 statements batch into one call.

**83 statements, 48% positive.** Class balance is deliberate — κ becomes
unstable when one class dominates, and on a skewed set a judge scores well
by always answering the majority class.

Contexts are **real**: abstracts from `eval/fixtures.py`, the same text the
pipeline indexes. Synthetic contexts would make the task cleaner than the
job the judge actually does.

Case types, chosen so most of them discriminate:

| type | n | what it catches |
|---|---|---|
| `supported` | 29 | over-strict judges |
| `paraphrase` | 11 | judges demanding matching vocabulary |
| `true_but_absent` | 8 | **judges answering "is this true?" instead of "does the passage support it?"** |
| `corrupted_entity` | 7 | wrong subject, right shape |
| `corrupted_number` | 6 | 94% → 84% — the most realistic RAG failure |
| `overgeneralised` | 6 | scope qualifier silently dropped |
| `unsupported_cause` | 6 | a "because" the passage never gives |
| `unrelated` | 5 | the floor |
| `negation` | 5 | claim reversed |

`true_but_absent` is the sharpest. A statement can be perfectly true and
still unsupported by *this* passage, and a judge that rewards world
knowledge is measuring the wrong thing while looking confident.

### Experiment 2 — agreement and stability (Spearman ρ)

Full ragas faithfulness over 15 real answers, **frozen** — generated once,
so every judge scores identical inputs.

- **ρ vs `claude-haiku-4.5`** — measures *ledger comparability*, explicitly
  not correctness. "Matches my previous judge" and "is a good judge" are
  different claims and only one is about quality.
- per-item σ across repeats — stability
- `NaN` / parse-failure rate

Both κ and ρ get reported, side by side, so the deflation gap is visible on
our own data rather than taken on faith from the paper.

### Three repeats, three jobs

| repeat | statement order | isolates |
|---|---|---|
| 1 | canonical | baseline |
| 2 | canonical | determinism at temperature 0 |
| 3 | **shuffled** | order sensitivity — the pointwise analogue of position bias |

Repeat 3 costs nothing extra and covers the consistency–bias paradox.

## Ground truth

**We had no faithfulness labels.** `relevant_source_ids` and the feedback
thumbs (#34) are *relevance* judgments — a different question.

Labels come from two places, and the distinction is worth keeping straight:

1. **By construction.** A statement asserting a number the passage doesn't
   contain is unsupported; that's arithmetic, not opinion.
2. **Human-verified.** `eval/labelling/index.html` presents each
   (passage, statement) pair for a human verdict.

Step 2 exists because step 1 alone is *my* say-so, and I wrote the cases —
I could have made them accidentally easy. Verification converts "labels by
construction" into "labels by construction, independently confirmed".

**The page withholds both `proposed_label` and `case_type`.** Showing either
would anchor the labeller to my answer, and a case type called
`corrupted_number` answers its own question. They are re-joined by `id` at
analysis time.

`agreement_with_proposed()` reports how far the human moved my labels, and
that number gets published. A high disagreement rate doesn't mean the
labeller erred — it means my cases were sloppier than claimed, and the fix
is the cases, not quietly keeping the labels I preferred.

Unsure verdicts are **dropped, not coerced**. An item nobody could decide
isn't ground truth, and guessing puts noise into exactly the numbers this
exercise exists to make trustworthy.

## Labelling outcome

All 83 labelled, 2 marked unsure and dropped, **81 usable at 43% positive**
— still inside the balance kappa needs.

**Agreement with my constructed labels: 95.1% (77/81).** The four
disagreements all run the same way — I said supported, the labeller said
not — and in all four the labeller is right or defensibly right:

| case | type | why my label was wrong |
|---|---|---|
| `c003`, `c004` | supported | Both drop *"on flat ground"*. The labelling instructions I wrote say dropping a scope qualifier makes a statement unsupported. **My construction contradicted my own rule.** |
| `c009` | paraphrase | "roughly nineteen of every twenty" for 94% asks the reader to do arithmetic the passage never does. |
| `c061` | paraphrase | My wording said the review compares *"ways of organising a control hierarchy"*; the passage lists hierarchical as one of four architectures surveyed. The paraphrase misreads it. |

This is precisely what the verification step was for. Had I scored judges
against my own labels, two of the eighty-one would have been graded against
a rule I had written down and then broken.

**Human labels stand as ground truth.** The four are additionally recorded
in `CONTESTED` and excluded from a sensitivity figure reported alongside the
primary kappa — a case two careful readers could split on measures the
case's ambiguity, not the judge's skill. A test asserts `CONTESTED` matches
the actual disagreements, so the two cannot drift apart if a case is edited.

## Decision rule — fixed before any data is seen

1. **Disqualify**: `NaN` rate > 5% · fails the `corrupted_number` cases · is
   the generator model
2. **Require**: κ ≥ 0.6 vs human labels · per-item σ < 0.1 · ρ ≥ 0.9 vs
   Haiku
3. **Then cheapest wins.** Cost breaks ties; it never overrides 1 or 2.

κ ≥ 0.6 is "substantial" on the standard scale and is deliberately stricter
than the 0.3–0.5 the large study found typical. If nothing clears it, that
is a finding to report, not a bar to lower.

**`minimax-m2.7:free` is excluded** despite being free and having scored
correctly in the first probe: it is the pipeline model, and self-preference
bias is measured at ~10% (GPT-4) to ~25% (Claude-v1) in the literature. It
would inflate precisely the metric this project has a known problem with —
agentic faithfulness 0.564 against the fixed baseline's 0.829.

## Cost

Four candidates. Haiku appears as the comparability reference only —
Experiment 2, single repeat — since its own stability isn't a decision
input.

| judge | calls | est. |
|---|---|---|
| `openai/gpt-oss-120b` | 150 | $0.014 |
| `deepseek/deepseek-v4-flash` | 150 | $0.025 |
| `qwen/qwen3-235b-a22b-2507` | 150 | $0.031 |
| `anthropic/claude-haiku-4.5` *(ref)* | 30 | $0.105 |
| **total** | **~480** | **≈ $0.18** |

Hard cap **$0.30**, abort if exceeded. Haiku is 60% of the bill at 15× the
token price — the thing we're trying to stop paying for, measured one last
time.

Ongoing once chosen: **~$0.035** per 50-question judged run against Haiku's
**~$0.525**. The experiment pays for itself on the first run.

## Status

**Built** — cases, labelling UI, label loader, tests (376 passing):

| file | |
|---|---|
| `eval/judge_cases.py` | 83 labelled statements, 9 case types, `load_labels()`, `agreement_with_proposed()` |
| `eval/labelling/build.py` | generates the page; withholds the answers |
| `eval/labelling/index.html` | self-contained labelling UI |
| `tests/test_judge_cases.py` | uniqueness, class balance, discriminating types present, unsure-dropping, disagreement reporting |

**Not built** — pending labels and approval to spend:

| file | |
|---|---|
| `eval/judges.py` | the harness: both experiments, 3 repeats, temp 0, cache asserted off |
| `eval/pipeline.py` | `StoredPipeline` (~10 lines) to replay frozen answers |
| `orchestrator/llm.py` | `temperature` parameter — **currently unset, so the judge runs at ChatOpenAI's default 0.7.** A bug independent of this experiment. |

## Limits, stated up front

- **83 statements is below the published range.** Constitutional AI used 254
  conversations; a factual-inconsistency study 373. This gives a directional
  signal with wide error bars on κ, not a definitive ranking.
- **One labeller, no inter-annotator agreement.** The usual target is
  LLM-human agreement approaching human-human agreement; with a single
  annotator there is no human-human baseline to compare against.
- **Constructed cases, not sampled from real failures.** They test the
  failure modes I anticipated. A judge could pass all of them and still miss
  something the pipeline actually does.

## Running it

```bash
cd services/backend
uv run python -m eval.labelling.build          # regenerate after editing cases
xdg-open eval/labelling/index.html             # ~35 min, saves as you go
# save the export to eval/labelling/labels.json
```

Progress is written to localStorage after every answer and the page resumes
at the first unlabelled statement, so the tab can be closed at will.
**Export progress** is available at any point, not only on completion — a
partial file is valid input, since `load_labels()` scores over whatever
labels exist. If the browser blocks local storage (a private window, or
"clear site data on close"), the page says so in red rather than losing the
work silently.
