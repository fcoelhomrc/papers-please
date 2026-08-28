"""Runs a Pipeline against eval/dataset.jsonl and scores it with Ragas.

Costs real LLM judge calls (one call per metric per question, roughly) -
not wired into pytest/CI. Run on demand:

    uv run python -m eval.run --variant fixed
    uv run python -m eval.run --variant agentic

Judge LLM is Claude (via ragas' LangchainLLMWrapper around our own
make_llm()) - no OpenAI key needed, consistent with the rest of the stack.
Judge embeddings reuse the same bge-small model already used for search
(wrapped for ragas via langchain_community's HuggingFaceEmbeddings) - local,
no extra API cost.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from eval.pipeline import AgenticPipeline, FixedPipeline, Pipeline

RESULTS_DIR = Path(__file__).parent / "results"
DATASET_PATH = Path(__file__).parent / "dataset.jsonl"

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_eval(pipeline: Pipeline, dataset_path: Path, variant_name: str, judge_llm, judge_embeddings) -> dict:
    rows = load_dataset(dataset_path)
    records = []
    for row in rows:
        result = pipeline.answer(row["question"])
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
    eval_result = evaluate(
        dataset,
        metrics=METRICS,
        llm=LangchainLLMWrapper(judge_llm),
        embeddings=LangchainEmbeddingsWrapper(judge_embeddings),
    )

    df = eval_result.to_pandas()
    per_question = df.to_dict(orient="records")
    means = {m.name: float(df[m.name].mean()) for m in METRICS if m.name in df.columns}

    output = {"variant": variant_name, "means": means, "per_question": per_question}

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{variant_name}-{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))

    return output


def _build_pipeline(variant: str) -> Pipeline:
    from config import load
    from orchestrator.graph import build_agent
    from orchestrator.llm import make_llm
    from search import get_search_engine

    llm = make_llm(load())
    if variant == "fixed":
        return FixedPipeline(llm, get_search_engine())
    if variant == "agentic":
        return AgenticPipeline(build_agent(llm))
    raise ValueError(f"unknown variant: {variant!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["fixed", "agentic"], required=True)
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    args = parser.parse_args()

    from config import load
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from orchestrator.llm import make_llm
    from process.embedder import MODELS

    cfg = load()
    judge_llm = make_llm(cfg)
    judge_embeddings = HuggingFaceEmbeddings(model_name=MODELS[cfg.embedder.model]["hf_name"])

    pipeline = _build_pipeline(args.variant)
    output = run_eval(pipeline, Path(args.dataset), args.variant, judge_llm, judge_embeddings)

    print(f"variant: {output['variant']}")
    for name, score in output["means"].items():
        print(f"  {name}: {score:.3f}")


if __name__ == "__main__":
    main()
