"""The judged branch: generate answers, then score them with an LLM judge.

MANUAL ONLY, and it spends real money.

    uv run python -m eval.run --sample 10        # iterate cheaply
    uv run python -m eval.run                    # full 100-question run

Roughly $0.12 for the full set. Never invoked from compose.yaml, from a stage
worker, or from anything automatic - it runs when a person types the command.

What a judge is for, and what it is not for
-------------------------------------------
The free branch (eval/run_free.py) already scores retrieval exactly, against
chunk-id labels, for zero tokens. It is strictly better than a judge at that
job: it compares against ground truth rather than against a judge's opinion of
ground truth. So nothing here exists to re-measure ranking.

What only a judge can do is read generated prose:

  faithfulness       is every claim in the answer supported by the retrieved
                     context - the hallucination detector
  response_relevancy does the answer address the question that was asked
  llm_context_precision  were the useful chunks ranked first, judged by
                     usefulness rather than by matching a label
  llm_context_recall did retrieval get everything the gold answer needed

The last two overlap with the free branch on purpose. They ask the *same*
question against different ground truth: the free branch asks "did you find
the chunk this question was written from", the judge asks "did you find
something that answers it". Where they disagree, the free branch is usually
the pessimistic one - the seed chunk is a sample of the relevant chunks, not
the complete set, so a correct retrieval of a different chunk scores as a miss.
Reading the two together is the point; reading either alone is not.

Why this matters for the headline result
----------------------------------------
The free branch's strongest finding is that BM25 beats dense retrieval. That
finding carries a known bias: the questions were generated *from* the chunks,
so they inherit chunk vocabulary in a way a real user's phrasing would not,
and lexical matching is exactly what benefits. A judge scoring answer quality
does not care which chunk the words came from. This branch is the check on
that result, not a formality.

Abstention is reported separately
---------------------------------
`ResponseRelevancy` scores a noncommittal answer 0. A pipeline that correctly
says "the library has nothing on this" is therefore punished by it, and
letting that sit inside the mean would make honesty look like failure.
"""
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.cache import DiskCacheBackend
from ragas.cost import (
    TokenUsage,
    get_token_usage_for_anthropic,
    get_token_usage_for_openai,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from eval.pipeline import FixedPipeline, Pipeline
from eval.retrieval import RETRIEVAL_METRICS, score_question

from eval.results_store import encoder_for, results_dir, results_key
JUDGE_CACHE_DIR = Path(__file__).parent / ".judge-cache"

# The judge's own embedder, held fixed across every arm of an encoder sweep.
# bge-small is what every judged run so far has scored with, so pinning here
# keeps the runs already on disk comparable to the ones still to come.
JUDGE_EMBED_MODEL = "bge-small"

# All four, unlike the two this file used to run. The pair that was dropped -
# context precision and recall - was dropped because eval/sweep.py measured
# ranking for free against document-level labels. That reasoning no longer
# holds: the labels are chunk-level now and the free branch measures them
# exactly, so these two are no longer a weaker copy of something already
# measured. They are the judge's *different* answer to the same question, and
# the disagreement between the two is the finding.
#
# strictness=1 on relevancy (ragas' default is 3) generates one reverse
# question per answer instead of three. Averaging three buys stability in the
# third decimal, below the noise floor of a 100-question set.
METRICS = [
    Faithfulness(),
    ResponseRelevancy(strictness=1),
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
]

# Single source of truth in eval/pricing.py, aliased here so the two
# long-standing names keep working. Both the judged run and test-set
# generation report spend and must not disagree about the rates.
from eval.pricing import PRICING as JUDGE_PRICING  # noqa: E402
from eval.pricing import model_price as judge_price  # noqa: E402

# An answer this short and this hedged is an abstention, not a response.
# Deliberately crude: the point is to *separate* these rows, and a judge call
# to classify them would cost more than the metric they are being kept out of.
ABSTENTION_MARKERS = (
    "does not contain",
    "doesn't contain",
    "no information",
    "not enough information",
    "cannot answer",
    "can't answer",
    "no relevant",
    "not mentioned",
    "unable to answer",
)


def is_abstention(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(m in lowered for m in ABSTENTION_MARKERS)


def token_usage_parser(cfg):
    """The parser matching the judge's wire format.

    Ragas reads token counts out of the raw provider response, and the two
    shapes differ: Anthropic reports `usage.input_tokens`, OpenAI-compatible
    endpoints (OpenRouter included) report `token_usage.prompt_tokens`. Using
    the wrong one doesn't fail - it silently returns zeros, which would gut
    the cost reporting this harness exists to provide.
    """
    if cfg.llm.provider == "anthropic":
        return get_token_usage_for_anthropic
    return get_token_usage_for_openai


def stratified_sample(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """`n` questions spread across synthesizers, not the first `n`.

    The set is 46/33/21 single-hop / multi-hop-abstract / multi-hop-specific
    and those three score very differently, so an unstratified sample of 10
    reports whichever type it happened to draw.
    """
    if n >= len(rows):
        return rows
    buckets: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        buckets.setdefault(row["synthesizer"], []).append(i)

    rng = random.Random(seed)
    picked: set[int] = set()
    for name in sorted(buckets):
        share = round(n * len(buckets[name]) / len(rows))
        picked.update(rng.sample(buckets[name], min(share, len(buckets[name]))))
    # Per-bucket rounding can land a row or two short of n; top up
    # deterministically from whatever is left.
    if len(picked) < n:
        rest = [i for i in range(len(rows)) if i not in picked]
        picked.update(rng.sample(rest, min(n - len(picked), len(rest))))

    # Indices, then sorted back into dataset order: a sample that reorders the
    # set makes two runs harder to diff line by line for no benefit, and the
    # rows are returned as the original objects, not copies.
    return [rows[i] for i in sorted(picked)[:n]]


def judge_spend(eval_result, judge_model_name: str) -> dict:
    """What the judge cost, in tokens and (where the price is known) dollars.

    Ragas only tracks this when evaluate() was handed a token_usage_parser and
    raises otherwise, so a differently-configured run reports nothing rather
    than failing. Accounting must never break a run whose calls are paid for.
    """
    try:
        usage = eval_result.total_tokens()
    except Exception as e:
        print(f"warning: no judge token usage recorded ({e})")
        return {}

    if isinstance(usage, list):
        usage = sum(usage, TokenUsage(input_tokens=0, output_tokens=0))

    spend = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
    price = judge_price(judge_model_name)
    if price is not None:
        spend["usd"] = round(usage.cost(*price), 4)
    return spend


def score_retrieval(rows: list[dict], retrieved: list[list[int]]) -> dict:
    """The free branch's exact chunk-id metrics, over what this run retrieved.

    Reported beside the judged scores because the judge cannot separate
    "retrieval never found it" from "retrieval found it and the model ignored
    it" - faithfulness drops either way, and the fix is different.
    """
    from eval.run_free import mean_ci

    per_question = [
        score_question(chunks, set(row["reference_chunk_ids"]), k=max(len(chunks), 1))
        for row, chunks in zip(rows, retrieved)
        if row["reference_chunk_ids"]
    ]
    answerable = [q for q in per_question if not q["abstention"]]
    if not answerable:
        return {}
    out = {"n": len(answerable)}
    for metric in RETRIEVAL_METRICS:
        mean, half = mean_ci([q[metric] for q in answerable])
        out[metric] = round(mean, 4)
        out[f"{metric}_ci"] = round(half, 4)
    return out


def answer_all(pipeline: Pipeline, rows: list[dict]) -> tuple[list[dict], list[list[int]], int]:
    """Run the pipeline over every question, surviving individual failures.

    One question's failure must never take the other N-1 down: a real incident
    had a single question hit LangGraph's recursion limit and crash the loop
    before anything reached disk, discarding ~40 already-paid-for answers.
    Catch, record the failure as the answer, keep going.
    """
    records, retrieved = [], []
    empty = failed = 0
    for i, row in enumerate(rows, 1):
        try:
            result = pipeline.answer(row["question"])
        except Exception as e:
            print(f"[{i}/{len(rows)}] FAILED: {row['question'][:60]!r} - {e}")
            failed += 1
            result = {"answer": f"error: pipeline failed ({e})", "contexts": [], "chunk_ids": []}
        else:
            # An empty answer is a silent failure, and it looks exactly like a
            # successful run until the metrics come back: faithfulness has no
            # statements to decompose (nan) and relevancy has no answer to
            # reverse-question (0.0). The cause is usually the answerer, not
            # the judge - a reasoning model spending its whole budget on
            # reasoning, or PAPERS_PLEASE_REPLAY leaving make_llm on the
            # recorded-cassette model, which returns "" for anything it has no
            # recording of. Say so at the point it happens.
            if not (result.get("answer") or "").strip():
                empty += 1
                print(f"[{i}/{len(rows)}] EMPTY ANSWER: {row['question'][:50]!r}")
            else:
                print(f"[{i}/{len(rows)}] ok: {row['question'][:60]!r}")

        retrieved.append(result.get("chunk_ids") or [])
        records.append(
            {
                "user_input": row["question"],
                "response": result["answer"],
                # ragas requires non-empty retrieved_contexts even when
                # retrieval genuinely found nothing; an explicit placeholder
                # beats crashing the run.
                "retrieved_contexts": result["contexts"] or ["(no context retrieved)"],
                "reference": row["reference"],
            }
        )

    # A run that lost its network partway through still *completes*: every
    # question after the drop records its exception as the answer, the judge
    # scores those strings, and the result file looks structurally valid. It
    # would then win judged_by_arm()'s newest-run-per-arm and silently replace
    # a good run in the figures. One or two failures are the per-question
    # resilience working as intended; a tenth of the set is a broken run.
    if failed > max(2, len(rows) // 10):
        raise RuntimeError(
            f"{failed}/{len(rows)} questions failed - refusing to write a result "
            f"file that would look valid. Usually a dropped connection partway "
            f"through; re-run this arm."
        )
    if empty:
        # Refuse rather than spend on judging blanks. Every generation metric
        # is undefined against an empty response, so the run would cost full
        # price and report nan.
        raise RuntimeError(
            f"{empty}/{len(rows)} answers came back empty - judging these would "
            f"cost full price for nan. Check PAPERS_PLEASE_REPLAY is unset and "
            f"that llm.model is not a reasoning model whose max_tokens is too "
            f"small to leave room for an answer."
        )
    return records, retrieved, failed


def run_eval(
    pipeline: Pipeline,
    rows: list[dict],
    judge_llm,
    judge_embeddings,
    model_name: str = "",
    judge_model_name: str = "",
    prompt_versions: dict[str, str] | None = None,
    retrieval: dict | None = None,
    usage_parser=get_token_usage_for_openai,
) -> dict:
    records, retrieved, failed = answer_all(pipeline, rows)

    # Memoised by a hash of the prompt, so re-running an unchanged set (a
    # report tweak, a crash after answers were generated) re-reads instead of
    # re-paying. Judge calls only - the pipeline's own answers are not cached,
    # since the point of a run is usually that the pipeline changed.
    cache = DiskCacheBackend(cache_dir=str(JUDGE_CACHE_DIR))
    eval_result = evaluate(
        EvaluationDataset.from_list(records),
        metrics=METRICS,
        llm=LangchainLLMWrapper(judge_llm, cache=cache),
        embeddings=judge_embeddings,
        # Without this ragas records no usage at all and total_tokens()
        # raises. A run that cannot say what it cost is how you end up
        # guessing at the bill instead of reading it.
        token_usage_parser=usage_parser,
    )

    df = eval_result.to_pandas()
    per_question = df.to_dict(orient="records")
    names = [m.name for m in METRICS if m.name in df.columns]

    # How many questions each metric actually scored. A judge call that ran out
    # of output budget is recorded by ragas as NaN, and pandas' .mean() skips
    # NaN - so a metric scored on half the set reads as a clean number with
    # nothing to say it was halved. Coverage rides beside every mean, because
    # two metrics scored over different subsets are not comparable.
    coverage = {n: int(df[n].notna().sum()) for n in names}

    abstained = [is_abstention(r["response"]) for r in records]
    means = {n: float(df[n].mean()) for n in names}
    # Relevancy scores a noncommittal answer 0, so a correct abstention drags
    # the mean down for behaving well. The answering subset is reported beside
    # the full mean rather than instead of it.
    answered = df[[not a for a in abstained]]
    means_answered = {n: float(answered[n].mean()) for n in names} if len(answered) else {}

    mode = (retrieval or {}).get("mode")
    embed_model = encoder_for(mode)

    output = {
        "kind": "judged",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "means": means,
        "coverage": coverage,
        "means_excluding_abstentions": means_answered,
        "n_abstentions": sum(abstained),
        "n_failed": failed,
        "n_questions": len(rows),
        # None for a keyword-only or BM25 run: those rank on Postgres text and
        # never call an encoder, so naming one would claim a comparison the run
        # did not make.
        "embed_model": embed_model,
        "answerer_model": model_name,
        "judge_model": judge_model_name,
        "judge_spend": judge_spend(eval_result, judge_model_name),
        "prompt_versions": prompt_versions or {},
        "retrieval_config": retrieval or {},
        # Judge-free retrieval scores alongside the judged generation scores:
        # they separate "retrieval never found it" from "retrieval found it and
        # the model didn't use it", which the judged metrics conflate.
        "retrieval_metrics": score_retrieval(rows, retrieved),
        "per_question": per_question,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    arm = (retrieval or {}).get("arm", "none")
    path = results_dir(results_key(mode)) / f"judged-{arm}-{stamp}.json"
    path.write_text(json.dumps(output, indent=2, default=str))
    output["results_path"] = str(path)
    return output


def build_pipeline(versions: dict[str, str], arm: str | None = None,
                   mode: str | None = None) -> tuple[Pipeline, str]:
    """The fixed baseline only.

    The agentic arm is deliberately not evaluated: what it added over this was
    a query string and a stop decision, and the query-side techniques that
    actually matter now live in eval/query_arms.py as measurable retrieval
    arms rather than inside an opaque loop.
    """
    from config import load
    from orchestrator.llm import make_llm
    from prompts.registry import load_prompt
    from search import get_search_engine

    cfg = load()
    llm = make_llm(cfg)
    prompt = load_prompt("fixed_rag", versions["fixed_rag"])

    if arm or mode:
        from eval.pipeline import ArmPipeline
        from eval.query_arms import queries_for
        from eval.review import load_curated

        arm = arm or "none"
        rows = load_curated()
        # `none` routes through the same path as every arm rather than through
        # FixedPipeline, which reads the configured mode and would make the
        # baseline a different retriever from the arms it is the baseline for.
        by_id = queries_for(arm, rows)
        # Keyed by question text, because the pipeline is handed a question and
        # not a row id.
        queries = {r["question"]: by_id[r["id"]] for r in rows if r["id"] in by_id}
        return ArmPipeline(
            llm, get_search_engine(), prompt, arm, queries,
            mode or cfg.search.mode, top_k=cfg.search.top_k,
        ), cfg.llm.model

    pipeline = FixedPipeline(
        llm,
        get_search_engine(),
        system_prompt=prompt,
        top_k=cfg.search.top_k,
        rerank=False,
        candidates=None,
    )
    return pipeline, cfg.llm.model


# 2048 -> 8192 -> 32768. Faithfulness emits a verdict with a reason per statement,
# against every retrieved context at once, so its output grows with both the
# answer's length and top_k. At 2048 it raised LLMDidNotFinishException, which
# ragas records as nan rather than as an error - the metric simply comes back
# empty while every other metric scores normally, which reads as a broken
# judge instead of a truncated one. Cost is bounded by what is emitted, not by
# this ceiling. 8192 was still not enough: an audit found faithfulness nan on
# 51/100 of the first full run and 18-21/100 of two arms. Because pandas skips
# NaN in .mean(), the metric read as a clean number computed over half the set,
# and the arms producing the longest answers truncated most often - so the
# means were not comparable across arms either.
JUDGE_MAX_TOKENS = 32768


def judge_llm(cfg):
    from orchestrator.llm import openrouter_chat

    model = cfg.llm.judge_model
    if cfg.llm.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, max_tokens=JUDGE_MAX_TOKENS), model
    return openrouter_chat(model, max_tokens=JUDGE_MAX_TOKENS, cfg=cfg), model


def judge_embeddings():
    """bge-small locally, pinned - not whatever the retriever happens to be.

    ResponseRelevancy needs embeddings to compare its reverse-generated
    questions against the original; paying an API for that would be the
    largest line on the bill for the least interesting part of it.

    Pinned because this used to read `embedder.model`, which meant the judge
    measured answer relevancy with a different yardstick in every arm of an
    encoder sweep - a confound in precisely the comparison the sweep exists to
    make, and one that moves the metric without touching answer quality. It
    also crashed outright on an encoder needing trust_remote_code, since this
    path does not go through load_encoder().

    The judge's embedder is part of the judge, not part of what is being
    judged, so it stays fixed while the retrieval encoder varies.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from process.embedder import MODELS

    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=MODELS[JUDGE_EMBED_MODEL]["hf_name"])
    )


def fmt(output: dict) -> None:
    print(f"\n{output['n_questions']} questions | answerer={output['answerer_model']} "
          f"| judge={output['judge_model']}")
    print(f"retrieval: {output['retrieval_config']}\n")

    n = output["n_questions"]
    print("judged (generation):")
    for name, value in sorted(output["means"].items()):
        scored = output.get("coverage", {}).get(name, n)
        mark = "" if scored == n else f"   <- scored {scored}/{n}, NOT comparable"
        print(f"  {name:<34} {value:.3f}   n={scored}{mark}")
    print(f"  {'abstentions':<34} {output['n_abstentions']}")
    if output.get("n_failed"):
        print(f"  {'pipeline failures':<34} {output['n_failed']}")

    if output["retrieval_metrics"]:
        print("\nretrieval (chunk-id labels, no judge):")
        r = output["retrieval_metrics"]
        print("  " + "  ".join(f"{m}={r[m]:.3f}" for m in RETRIEVAL_METRICS if m in r))

    spend = output["judge_spend"]
    if spend:
        usd = f" = ${spend['usd']}" if "usd" in spend else ""
        print(f"\njudge spend: {spend['input_tokens']:,} in + "
              f"{spend['output_tokens']:,} out{usd}")
    print(f"\n-> {output['results_path']}")


def main():
    parser = argparse.ArgumentParser(description="Judged eval run (costs money)")
    parser.add_argument("--sample", type=int, default=None, help="stratified subset")
    parser.add_argument("--arm", default=None,
                        help="query arm to retrieve through (default: none)")
    parser.add_argument("--mode", default=None, help="retrieval mode override")
    parser.add_argument("--prompt-version", action="append", default=None,
                        help="name=version, e.g. fixed_rag=v1")
    args = parser.parse_args()

    from config import load
    from eval.review import load_curated
    from observability import setup_observability

    setup_observability("eval-judged")

    cfg = load()

    # Before anything expensive, and well before the first judge token: an arm
    # that cannot run in this mode must fail here rather than silently produce
    # a run that measures something other than its filename claims. HyDE under
    # --mode bm25 took the dense path regardless and landed in results/bm25/.
    # Only checked on the ArmPipeline path, which is the one build_pipeline
    # takes when either flag is given.
    if args.arm or args.mode:
        from eval.query_arms import modes_for, supports

        arm = args.arm or "none"
        mode = args.mode or cfg.search.mode
        if not supports(arm, mode):
            raise SystemExit(
                f"arm {arm!r} cannot run in mode {mode!r}; "
                f"it supports {', '.join(modes_for(arm))}"
            )

    versions = {"fixed_rag": cfg.prompts.fixed_rag}
    for override in args.prompt_version or []:
        name, _, version = override.partition("=")
        versions[name] = version

    rows = load_curated()
    if args.sample:
        rows = stratified_sample(rows, args.sample)

    pipeline, answerer = build_pipeline(versions, arm=args.arm, mode=args.mode)
    judge, judge_model = judge_llm(cfg)

    print(f"answering {len(rows)} questions with {answerer}, judging with {judge_model}")
    output = run_eval(
        pipeline,
        rows,
        judge,
        judge_embeddings(),
        model_name=answerer,
        judge_model_name=judge_model,
        prompt_versions=versions,
        retrieval={"mode": args.mode or cfg.search.mode, "top_k": cfg.search.top_k,
                   "rerank": False, "arm": args.arm or "none"},
        usage_parser=token_usage_parser(cfg),
    )
    fmt(output)


if __name__ == "__main__":
    main()
