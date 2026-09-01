"""Runs a Pipeline against eval/dataset.jsonl and scores it with Ragas.

MANUAL ONLY. Costs real Anthropic API calls (your ANTHROPIC_API_KEY - Ragas
has no key of its own, it just calls whatever LLM object you hand it): one
call per question for the pipeline's own answer, plus several judge calls
per metric per question. This is never invoked automatically - not from
compose.yaml, not from any CI workflow (there isn't one), not from any
stage worker. It only runs when a human types the command below.

    uv run python -m eval.run --variant fixed --sample 15   # iterate
    uv run python -m eval.run --variant agentic             # full, for the ledger

This is the *expensive* tier. The judge-free retrieval scores (eval.sweep,
eval.thresholds) measure ranking quality for zero tokens and are the loop to
run on every retrieval change; reach for this one only when you need to know
what the generated answer looks like, which is the one thing a judge adds.

Three things keep the bill down (#26): only two judged metrics rather than
four (see METRICS), --sample N for a stratified subset, and judge calls
memoised on disk so re-running an unchanged dataset re-reads instead of
re-paying.

Judge LLM is Claude (via ragas' LangchainLLMWrapper) - no OpenAI key
needed, consistent with the rest of the stack. Judge embeddings reuse the
same bge-small model already used for search (wrapped for ragas via
langchain_community's HuggingFaceEmbeddings) - local, no extra API cost.

Every run writes two artifacts:
  - eval/results/{variant}-{timestamp}.json - full raw output (gitignored,
    scratch/debugging use)
  - eval/reports/{variant}-{timestamp}.md - human-readable markdown report
    with summary + per-question tables (NOT gitignored - meant to be
    committed and reviewed)
"""
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from ragas import EvaluationDataset, evaluate
from ragas.cache import DiskCacheBackend
from ragas.cost import TokenUsage, get_token_usage_for_anthropic
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, faithfulness

from eval.pipeline import AgenticPipeline, FixedPipeline, Pipeline
from eval.report import write_markdown_report
from eval.retrieval import aggregate, score_question

RESULTS_DIR = Path(__file__).parent / "results"
DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
JUDGE_CACHE_DIR = Path(__file__).parent / ".judge-cache"

# Two judged metrics, not four. context_precision and context_recall were
# dropped in #26: eval/sweep.py already scores precision/recall/nDCG/MRR
# against the dataset's `relevant_source_ids` labels for zero tokens, and
# does it against ground truth rather than against a judge's opinion of
# ground truth. Paying for a weaker measurement of something already
# measured exactly was ~55% of the judge bill - context_precision alone
# issues one call *per retrieved context*, so its cost scaled with k.
#
# What's left is the pair a judge is genuinely required for, because both
# are properties of generated prose that no label can capture:
#   faithfulness      - is the answer supported by the retrieved context
#   answer_relevancy  - does the answer address the question that was asked
#
# strictness=1 (ragas' default is 3) generates one reverse-question per
# answer instead of three. Averaging over three buys stability in the third
# decimal, which is well below the noise floor of a 50-question set.
METRICS = [faithfulness, AnswerRelevancy(strictness=1)]

# USD per (input, output) token - Anthropic list prices, for turning the
# judge's token count into a number that means something. Only the models
# this project would plausibly judge with; an unlisted model records tokens
# and omits the cost rather than inventing a price.
JUDGE_PRICING = {
    "claude-haiku-4-5": (1.00 / 1e6, 5.00 / 1e6),
    "claude-sonnet-5": (2.00 / 1e6, 10.00 / 1e6),
    "claude-opus-5": (5.00 / 1e6, 25.00 / 1e6),
}


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_sample(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """`n` questions that keep the dataset's category x domain mix.

    A judged run is the expensive tier, so iterating on it means running a
    subset - but a subset drawn uniformly at random is a worse instrument
    than a smaller one drawn carefully. The dataset is deliberately built out
    of strata (grounded/edge_case x domain), and the edge cases are the rows
    that catch abstention regressions, so a sample that loses them measures
    the easy half of the problem and reports it as the whole.

    Round-robin across strata rather than a proportional allotment: with 50
    questions over ~8 strata, proportional rounding drives the small strata
    to zero, which is exactly the coverage a subset must not lose.

    Seeded, so `--sample 15` names the same 15 questions on every run and two
    scores taken a week apart stay comparable.
    """
    if n >= len(rows):
        return rows

    strata: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row.get("category", ""), row.get("domain") or row.get("subtype") or "")
        strata.setdefault(key, []).append(row)

    rng = random.Random(seed)
    for group in strata.values():
        rng.shuffle(group)

    picked: list[dict] = []
    order = sorted(strata)
    while len(picked) < n:
        drew = False
        for key in order:
            if not strata[key]:
                continue
            picked.append(strata[key].pop())
            drew = True
            if len(picked) == n:
                break
        if not drew:  # every stratum exhausted (n > len(rows) can't happen, but be safe)
            break

    # Restore dataset order so the report's per-question table reads in the
    # same sequence as the file it came from.
    position = {id(r): i for i, r in enumerate(rows)}
    return sorted(picked, key=lambda r: position[id(r)])


def judge_spend(eval_result, judge_model_name: str) -> dict:
    """What the judge cost, in tokens and (where the price is known) dollars.

    Ragas only tracks this when evaluate() was handed a token_usage_parser,
    and raises otherwise - so an older or differently-configured run reports
    nothing rather than failing. Never let accounting break a run whose API
    calls are already paid for.
    """
    try:
        usage = eval_result.total_tokens()
    except Exception as e:
        print(f"warning: no judge token usage recorded ({e})")
        return {}

    # total_tokens() returns a list when more than one model was involved.
    if isinstance(usage, list):
        usage = sum(usage, TokenUsage(input_tokens=0, output_tokens=0))

    spend = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
    price = JUDGE_PRICING.get(judge_model_name)
    if price:
        spend["usd"] = round(usage.cost(*price), 4)
    return spend


def score_retrieval(rows: list[dict], retrieved_docs: list[list[int]]) -> dict:
    """Rank metrics over what each pipeline actually retrieved.

    Skipped silently when the dataset has no labels - an older dataset should
    still be scorable on the judged metrics rather than failing the run.
    """
    if not all("relevant_source_ids" in r for r in rows):
        return {}

    from db.connection import PostgresInterface
    from eval.sweep import _to_source_ids, doc_id_to_source_id

    id_map = doc_id_to_source_id(PostgresInterface.connect())
    per_question = []
    for row, doc_ids in zip(rows, retrieved_docs):
        retrieved = _to_source_ids([{"doc_id": d} for d in doc_ids], id_map)
        # k is however many the pipeline chose to retrieve - unlike the sweep,
        # this isn't a fixed budget, so precision is over what it actually used.
        per_question.append(
            score_question(retrieved, set(row["relevant_source_ids"]), k=max(len(retrieved), 1))
        )
    return aggregate(per_question)


def run_eval(
    pipeline: Pipeline,
    dataset_path: Path,
    variant_name: str,
    judge_llm,
    judge_embeddings,
    model_name: str = "",
    judge_model_name: str = "",
    prompt_versions: dict[str, str] | None = None,
    retrieval: dict | None = None,
    sample: int | None = None,
    judge_cache_dir: Path | None = None,
) -> dict:
    all_rows = load_dataset(dataset_path)
    rows = stratified_sample(all_rows, sample) if sample else all_rows
    records = []
    retrieved_docs: list[list[int]] = []
    for i, row in enumerate(rows, 1):
        # A single question's pipeline.answer() must never take the other
        # N-1 down with it - a real incident showed why: one question hit
        # LangGraph's recursion limit (an uncaught GraphRecursionError),
        # which crashed this whole loop before anything reached disk,
        # discarding ~40 other questions' worth of real, already-paid-for
        # answers. Catch, record the failure as the answer, keep going.
        try:
            result = pipeline.answer(row["question"])
        except Exception as e:
            print(f"[{i}/{len(rows)}] FAILED: {row['question'][:60]!r} - {e}")
            result = {"answer": f"error: pipeline failed ({e})", "contexts": [], "doc_ids": []}
        else:
            print(f"[{i}/{len(rows)}] ok: {row['question'][:60]!r}")

        retrieved_docs.append(result.get("doc_ids") or [])
        records.append(
            {
                "user_input": row["question"],
                "response": result["answer"],
                # ragas requires non-empty retrieved_contexts even when a
                # pipeline genuinely found nothing (e.g. the off-topic
                # question in the eval set) - an explicit "no context" beats
                # crashing the eval run.
                "retrieved_contexts": result["contexts"] or ["(no context retrieved)"],
                "reference": row["ground_truth"],
            }
        )

    dataset = EvaluationDataset.from_list(records)
    # Memoised by a hash of the prompt, so re-running an unchanged dataset
    # (a report tweak, a crash after the answers were generated, comparing a
    # report format) re-reads instead of re-paying. Only the *judge* calls -
    # the pipeline's own answers above are not cached, since the whole point
    # of a run is usually that the pipeline changed.
    cache = DiskCacheBackend(cache_dir=str(judge_cache_dir)) if judge_cache_dir else None
    eval_result = evaluate(
        dataset,
        metrics=METRICS,
        llm=LangchainLLMWrapper(judge_llm, cache=cache),
        embeddings=LangchainEmbeddingsWrapper(judge_embeddings),
        # Without this ragas records no usage at all and total_tokens() raises.
        # A run that can't say what it cost is how you end up guessing at a
        # 1M-token bill instead of reading it.
        token_usage_parser=get_token_usage_for_anthropic,
    )

    df = eval_result.to_pandas()
    per_question = df.to_dict(orient="records")
    means = {m.name: float(df[m.name].mean()) for m in METRICS if m.name in df.columns}

    output = {
        "variant": variant_name,
        "means": means,
        "n_questions": len(rows),
        "n_dataset": len(all_rows),
        "judge_spend": judge_spend(eval_result, judge_model_name),
        "per_question": per_question,
        "prompt_versions": prompt_versions or {},
        "retrieval": retrieval or {},
        # Judge-free retrieval scores alongside the judged generation scores:
        # they separate "retrieval never found it" from "retrieval found it
        # and the model didn't use it", which the Ragas metrics conflate.
        "retrieval_metrics": score_retrieval(rows, retrieved_docs),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{variant_name}-{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))

    report_path = write_markdown_report(output, rows, model_name, judge_model_name)
    output["report_path"] = str(report_path)

    # Bookkeeping must never fail a run that has already been paid for: by
    # this point every judge call is spent and the report is on disk, so a
    # ledger problem is a note to stderr, not an exception.
    try:
        from eval.ingest import parse_judged_report
        from eval.ledger import append

        record = parse_judged_report(report_path)
        if record and append(record):
            print("recorded in eval/ledger.jsonl")
    except Exception as e:
        print(f"warning: could not record run in the ledger ({e})")

    return output


def _build_pipeline(variant: str, versions: dict[str, str]) -> Pipeline:
    from config import load
    from orchestrator.graph import build_agent
    from orchestrator.llm import make_llm
    from prompts.registry import load_prompt
    from search import get_search_engine

    llm = make_llm(load())
    if variant == "fixed":
        prompt = load_prompt("fixed_rag", versions["fixed_rag"])
        return FixedPipeline(
            llm,
            get_search_engine(),
            system_prompt=prompt,
            candidates=load().search.rerank_candidates,
        )
    if variant == "agentic":
        return AgenticPipeline(build_agent(llm, version=versions["orchestrator"]))
    raise ValueError(f"unknown variant: {variant!r}")


def _resolve_prompt_versions(overrides: list[str] | None) -> dict[str, str]:
    """Config defaults, with `--prompt-version name=version` on top. Fails
    loudly on an unknown name: silently ignoring a typo'd override would
    produce a report that names a prompt version it did not actually run."""
    from config import load

    versions = load().prompts.model_dump()
    for item in overrides or []:
        name, _, version = item.partition("=")
        if not version:
            raise ValueError(f"--prompt-version expects name=version, got {item!r}")
        if name not in versions:
            raise ValueError(
                f"unknown prompt {name!r} (known: {', '.join(sorted(versions))})"
            )
        versions[name] = version
    return versions


def _judge_llm(cfg):
    # Ragas' judge needs more room than the chat agent's UX-tuned 512 - its
    # metrics prompt for reasoning before a verdict, and briefer completions
    # were hitting LLMDidNotFinishException (truncated before it could
    # finish). A separate, judge-specific construction rather than reusing
    # make_llm()'s max_tokens=512.
    return ChatAnthropic(model=cfg.llm.model, max_tokens=2048)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["fixed", "agentic"], required=True)
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument(
        "--prompt-version",
        action="append",
        metavar="NAME=VERSION",
        help="override a configured prompt version, e.g. orchestrator=v2 "
        "(repeatable). Recorded in the report so the score names its prompt.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="score a stratified N-question subset instead of the whole "
        "dataset. Seeded, so the same N is the same N every time. Use it "
        "while iterating; run the full set for a ledger entry.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="bypass the on-disk judge cache and re-pay for every judge call. "
        "Only needed if you suspect a stale cached verdict.",
    )
    args = parser.parse_args()

    from observability import setup_observability

    setup_observability(f"papers-please-eval-{args.variant}")

    from config import load
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from process.embedder import MODELS

    cfg = load()
    judge_llm = _judge_llm(cfg)
    judge_embeddings = HuggingFaceEmbeddings(model_name=MODELS[cfg.embedder.model]["hf_name"])

    # Makes eval reproducible regardless of the dev DB's current state (a
    # wipe, a fresh clone, whatever's been manually fetched) - the dataset's
    # questions are grounded in eval/fixtures.py, not in whatever happens
    # to already be ingested.
    from eval.seed import ensure_fixtures_seeded

    ensure_fixtures_seeded()

    # Resolved before any spend: a bad --prompt-version should fail here, not
    # after 50 questions' worth of paid API calls.
    versions = _resolve_prompt_versions(args.prompt_version)

    pipeline = _build_pipeline(args.variant, versions)
    output = run_eval(
        pipeline,
        Path(args.dataset),
        args.variant,
        judge_llm,
        judge_embeddings,
        model_name=cfg.llm.model,
        judge_model_name=cfg.llm.model,
        prompt_versions=versions,
        # Not a versioned prompt (see prompts/registry.py) but it does shape
        # every retrieval score, so a report should still name it.
        retrieval={
            "embed_model": MODELS[cfg.embedder.model]["hf_name"],
            "query_prompt": MODELS[cfg.embedder.model]["query_prompt"],
        },
        sample=args.sample,
        judge_cache_dir=None if args.no_cache else JUDGE_CACHE_DIR,
    )

    print(f"variant: {output['variant']}")
    for name, score in output["means"].items():
        print(f"  {name}: {score:.3f}")
    print(f"prompts: {', '.join(f'{k}={v}' for k, v in versions.items())}")
    spend = output.get("judge_spend") or {}
    if spend:
        cost = f" — ${spend['usd']:.4f}" if "usd" in spend else ""
        print(
            f"judge spend: {spend['input_tokens']:,} in / "
            f"{spend['output_tokens']:,} out{cost}"
        )
    print(f"report: {output['report_path']}")


if __name__ == "__main__":
    main()
