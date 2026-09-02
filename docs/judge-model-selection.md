# Picking an LLM judge

Tested 2026-09-02, after moving to OpenRouter (#18). Total cost of the
experiment: **$0.017**.

## Why this needed testing at all

The eval harness scores answers with an LLM judge (`faithfulness`,
`answer_relevancy`). Every judged number in `eval/ledger.jsonl` was produced
by `claude-haiku-4-5`, at roughly **$0.53 per 50-question run**.

OpenRouter makes the judge a config string, so the question became: is there
a cheaper model that judges *the same way*? "Cheaper" is easy — the list is
full of models at 3¢ a run. "The same way" is the part that needed evidence,
because a judge that is cheap and wrong is worth less than no judge.

Model metadata is not evidence. `supported_parameters` claiming
`structured_outputs` says nothing about whether ragas can parse what comes
back, and nothing at all about whether the verdict is correct.

## The test

Three faithfulness cases with known-correct verdicts, judged by ragas'
actual `Faithfulness` metric — so this exercises the real integration
(prompting, JSON parsing, NLI decomposition), not a proxy.

| case | answer | correct verdict |
|---|---|---|
| supported | restates the context's numbers | high (~1.0) |
| fabricated | true claim **plus** invented GPU-hours and a commercial deployment | middling |
| unsupported | a claim about a different field entirely | ~0.0 |

A usable judge must (a) complete without parse errors and (b) rank
supported > fabricated > unsupported. Script:
`scratchpad/judge_probe.py`.

## Results

Cost is per 50-question judged run at the current metric set (~150 judge
calls, ~300k in / 45k out).

| model | $/run | time | supported | fabricated | unsupported | verdict |
|---|---|---|---|---|---|---|
| `anthropic/claude-haiku-4.5` *(incumbent)* | 0.525 | 33.8s | 1.0 | 0.33 | 0.0 | ✅ reference |
| **`deepseek/deepseek-v4-flash`** | **0.035** | 63.0s | 1.0 | 0.33 | 0.0 | ✅ **chosen** |
| `qwen/qwen3-235b-a22b-2507` | 0.042 | 49.5s | 1.0 | 0.33 | 0.0 | ✅ |
| `minimax/minimax-m2.7:free` | 0.000 | 39.7s | 1.0 | 0.33 | 0.0 | ✅ but excluded — see below |
| `openai/gpt-5-nano` | 0.033 | 49.1s | 1.0 | **NaN** | 0.0 | ❌ `LLMDidNotFinishException` |
| `openai/gpt-oss-120b` | 0.019 | **10.1s** | 1.0 | 0.33 | **NaN** | ❌ `OutputParserException` |

**Four models returned scores identical to the incumbent** — 1.0 / 0.33 /
0.0, exactly. That is the result that mattered: swapping the judge does not
invalidate the existing ledger, because on these cases it does not change
the answer.

### The two failures

- **`gpt-oss-120b`** — fastest by a wide margin (10s vs 34-63s) and cheapest
  of the tested set, but threw `OutputParserException`: it returned JSON
  ragas couldn't parse. Ragas records that case as `NaN`, so the failure is
  a **silently missing score**, not a crash. That is worse than a loud
  error — a run would complete and quietly under-report.
- **`gpt-5-nano`** — `LLMDidNotFinishException`, i.e. truncated at
  `max_tokens=2048`. It's a reasoning model and the reasoning tokens ate the
  budget before the verdict. Probably fixable by raising the judge's
  `max_tokens`, but a judge that needs tuning to return an answer is a worse
  default than one that doesn't.

## Decision: `deepseek/deepseek-v4-flash`

**15× cheaper than the incumbent, with identical verdicts** on all three
cases. $0.035 vs $0.525 per 50-question run.

Two things that decided it over the alternatives:

**Not the free model, even though it scored the same.**
`minimax/minimax-m2.7:free` is the *pipeline* model — the one generating the
answers. A model grading its own output is self-preference bias, a
well-documented LLM-as-judge failure mode, and it would quietly inflate
exactly the metric this project has a known problem with (agentic
faithfulness 0.564 vs the fixed baseline's 0.829). Saving 3½ cents is not
worth making the headline number untrustworthy. This is why `llm.judge_model`
exists as a separate config key from `llm.model`.

**Not `gpt-oss-120b`, despite being cheaper and 6× faster.** Its failure
mode is a missing score rather than an error, which is the kind of thing
that corrupts a table without anyone noticing.

`qwen/qwen3-235b-a22b-2507` is the runner-up: 20% dearer, 20% faster, same
verdicts. Worth switching to if DeepSeek's latency becomes annoying — 63s
for three cases is the slowest of the set.

```yaml
# services/backend/config.yaml
llm:
  model: minimax/minimax-m2.7:free       # generates
  judge_model: deepseek/deepseek-v4-flash # scores — deliberately different
```

## What this does *not* establish

- **Run-to-run stability is unverified.** Each model was judged once. A
  judge that swings between runs is useless regardless of how it scored
  here, and that check was cut short to avoid spending more. The first real
  `--sample 15` run will show it for free — if scores move between two runs
  of an unchanged dataset, the judge is the suspect. (Note the disk cache
  will mask this: use `--no-cache` to actually test it.)
- **Three cases is a smoke test, not an agreement study.** It shows the
  judges don't disagree on obvious cases. It does not show they agree on
  borderline ones, which is where a judge actually earns its keep.
- **Cheaper untested options exist**: `openai/gpt-oss-20b` ($0.015),
  `qwen/qwen3.7-flash` ($0.015), `mistralai/mistral-small-24b` ($0.019),
  `qwen/qwen3-30b-a3b` ($0.023). Listed for completeness — none were run, so
  none are recommended.

## Re-running it

```bash
cd services/backend
PYTHONPATH=. uv run python scratchpad/judge_probe.py \
  "deepseek/deepseek-v4-flash" "qwen/qwen3-235b-a22b-2507"
```

Costs well under a cent per model. Worth repeating if the ledger ever shows
a judged score moving without a corresponding pipeline change.
