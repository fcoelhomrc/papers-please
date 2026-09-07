"""Retrieval evaluation with no LLM anywhere. Free, so run it on every change.

    uv run python -m eval.run_free                      # config defaults
    uv run python -m eval.run_free --mode hybrid --top-k 10 --rerank

Scores the curated question set on exact chunk ids. Because no splitter ever
ran during generation, each `reference_contexts` string is byte-identical to a
row in `chunks`, and `eval.testset.map_chunk_ids` resolved it to an integer at
generation time. So relevance here is set membership on integers - exact,
deterministic, and immune to a similarity threshold being wrong.

Two rules this file exists to enforce
-------------------------------------
**Score `ChunkResult.text`, never `.context`.** `search.neighbour_window`
glues neighbouring chunks onto a hit as `context`; that string is no longer
the retrieved chunk, so matching on it would credit retrieval for its
neighbours.

**Chunk size and embedding model are pinned.** `reference_chunk_ids` point at
rows in `chunks` as they existed when the test set was generated. Re-chunking
renumbers them, and every score here silently becomes meaningless rather than
wrong-looking. A chunking ablation belongs in the judged branch.

On confidence intervals
-----------------------
They need no extra work, which is worth stating because the opposite is
widely assumed. The variance that matters is *across questions* - would a
different 100 give a different mean - and that is estimated from the spread of
the per-question scores already computed: `SE = s/sqrt(n)`. Judge variance is
the thing that needs repeats, and there is no judge here.

At n=100 the 95% interval on a mean is roughly +/-0.05 to +/-0.10 depending on
spread, so two configurations inside a few points of each other are a tie.
Reporting the interval is what stops that being forgotten.
"""
import argparse
import json
import logging
import math
from pathlib import Path

from eval.retrieval import RETRIEVAL_METRICS, score_question

logger = logging.getLogger(__name__)

from eval.results_store import results_dir

# Reported alongside the chunk-id metrics, not instead of them.
RAGAS_METRICS = ("ragas_precision", "ragas_recall")


def mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float]:
    """Mean and the half-width of its 95% confidence interval.

    The sample standard deviation over questions, divided by sqrt(n). A
    single question has no spread to estimate from, so its interval is
    reported as 0 rather than a fabricated number.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, z * math.sqrt(variance) / math.sqrt(n)


def paired_diff_ci(a: list[float], b: list[float], z: float = 1.96) -> tuple[float, float]:
    """Mean of the per-question differences, and its interval.

    Paired rather than two independent means: both configurations answered
    the same questions, so question difficulty cancels. The interval is
    tighter than comparing two separate CIs, and it is the right test for
    "is A better than B" - two overlapping individual intervals do not imply
    the difference is indistinguishable from zero.
    """
    return mean_ci([x - y for x, y in zip(a, b)], z=z)


def ragas_non_llm(retrieved_texts: list[str], reference_contexts: list[str]) -> dict:
    """ragas' own string-matching metrics, for parity with the ecosystem.

    They answer a slightly different question from the chunk-id metrics
    beside them. Ours is set membership on integers - exact, and immune to a
    threshold. Ragas compares *strings* with Levenshtein at a 0.5 cutoff,
    which is what anyone else reporting these numbers is measuring, so the
    two are reported together: if they diverge, the difference is the
    threshold, not retrieval.
    """
    import asyncio

    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall

    sample = SingleTurnSample(
        retrieved_contexts=retrieved_texts or ["(nothing retrieved)"],
        reference_contexts=reference_contexts,
    )
    return {
        "ragas_precision": asyncio.run(
            NonLLMContextPrecisionWithReference()._single_turn_ascore(sample, None)
        ),
        "ragas_recall": asyncio.run(
            NonLLMContextRecall()._single_turn_ascore(sample, None)
        ),
    }


def retrieve(engine, question: str, cfg: dict) -> list[tuple[int, str]]:
    """Ranked (chunk_id, text) pairs for one question.

    `text` not `context`: neighbour expansion rewrites `context` to include
    chunks that were never retrieved.
    """
    response = engine.search(
        question,
        top_k=cfg["top_k"],
        rerank=cfg["rerank"],
        rerank_top_k=cfg.get("rerank_top_k") or cfg["top_k"],
        mode=cfg["mode"],
        candidates=cfg.get("candidates"),
        thresholds=cfg.get("thresholds"),
        # Explicitly off. Expansion is a generation-time affordance; leaving
        # it on would not change which chunks rank, but it makes `context`
        # misleading for anything that reads it.
        neighbour_window=0,
    )
    return [(r.chunk_id, r.text) for r in response.results]


def score(rows: list[dict], engine, cfg: dict) -> tuple[list[dict], dict]:
    """Per-question scores and their aggregate, with intervals."""
    per_question = []
    for row in rows:
        hits = retrieve(engine, row["question"], cfg)
        retrieved = [cid for cid, _ in hits]
        scored = score_question(retrieved, set(row["reference_chunk_ids"]), cfg["top_k"])
        if not scored["abstention"]:
            scored.update(
                ragas_non_llm([t for _, t in hits], row["reference_contexts"])
            )
        scored["id"] = row["id"]
        scored["n_relevant"] = len(row["reference_chunk_ids"])
        scored["k"] = cfg["top_k"]
        scored["topics"] = row["topics"]
        scored["synthesizer"] = row["synthesizer"]
        scored["n_retrieved"] = len(retrieved)
        per_question.append(scored)
    return per_question, summarise(per_question)


def summarise(per_question: list[dict]) -> dict:
    """Aggregate with a confidence interval on every mean."""
    answerable = [q for q in per_question if not q["abstention"]]
    out: dict = {"n": len(answerable)}
    for metric in RETRIEVAL_METRICS:
        mean, half = mean_ci([q[metric] for q in answerable])
        out[metric] = round(mean, 4)
        out[f"{metric}_ci"] = round(half, 4)
    for metric in RAGAS_METRICS:
        values = [q[metric] for q in answerable if metric in q]
        if values:
            mean, half = mean_ci(values)
            out[metric] = round(mean, 4)
            out[f"{metric}_ci"] = round(half, 4)

    # Precision@k is bounded by how many gold chunks a question has: with one
    # gold chunk, precision@10 cannot exceed 0.1. Reporting the raw figure
    # alone reads as failure when it may be near its ceiling, so the ceiling
    # is reported with it and `precision_of_max` is the fraction attained.
    ceiling = _mean_ceiling(answerable)
    if ceiling:
        out["precision_ceiling"] = round(ceiling, 4)
        out["precision_of_max"] = round(out["precision"] / ceiling, 4) if ceiling else 0.0
    return out


def _mean_ceiling(answerable: list[dict]) -> float:
    """Mean of min(gold, k)/k - the best precision@k the set allows."""
    vals = [
        min(q["n_relevant"], q["k"]) / q["k"]
        for q in answerable
        if q.get("n_relevant") and q.get("k")
    ]
    return sum(vals) / len(vals) if vals else 0.0


def by_group(per_question: list[dict], key: str) -> dict[str, dict]:
    """The same aggregate, sliced.

    Reported alongside the mean because the set is not balanced: `agents`
    covers 58 of 100 questions, so the headline number is substantially a
    measure of that one topic. Thin slices - alignment at 8, evaluation at
    11 - carry intervals wide enough to be indicative only, which is exactly
    what the interval is there to show.
    """
    groups: dict[str, list[dict]] = {}
    for q in per_question:
        for value in q[key] if isinstance(q[key], list) else [q[key]]:
            groups.setdefault(value, []).append(q)
    return {name: summarise(qs) for name, qs in sorted(groups.items())}


def run(cfg: dict) -> dict:
    from datetime import datetime, timezone

    from config import load
    from eval.review import load_curated
    from search import get_search_engine

    rows = load_curated()
    engine = get_search_engine()
    logger.info(f"scoring {len(rows)} questions | {cfg}")

    per_question, overall = score(rows, engine, cfg)
    app = load()
    return {
        "kind": "retrieval_free",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        # Provenance, because these numbers are only comparable to others
        # measured on the same chunking and the same embedding model.
        "embed_model": app.embedder.model,
        "chunk_max_tokens": app.embedder.max_tokens,
        "corpus": app.search.corpus,
        "n_questions": len(rows),
        "overall": overall,
        "by_topic": by_group(per_question, "topics"),
        "by_synthesizer": by_group(per_question, "synthesizer"),
        "per_question": per_question,
    }


def fmt(name: str, summary: dict) -> str:
    parts = [
        f"{m}={summary[m]:.3f}+/-{summary[f'{m}_ci']:.3f}" for m in RETRIEVAL_METRICS
    ]
    if "precision_ceiling" in summary:
        parts.append(
            f"(prec ceiling {summary['precision_ceiling']:.3f}, "
            f"{summary['precision_of_max']:.0%} of max)"
        )
    for m in RAGAS_METRICS:
        if m in summary:
            parts.append(f"{m}={summary[m]:.3f}+/-{summary[f'{m}_ci']:.3f}")
    return f"  {name:<22} n={summary['n']:<4} " + "  ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Retrieval eval, no LLM calls")
    parser.add_argument("--mode", default=None, help="semantic | keyword | hybrid")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-top-k", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from config import load

    search_cfg = load().search
    cfg = {
        "mode": args.mode or search_cfg.mode,
        "top_k": args.top_k or search_cfg.top_k,
        "rerank": args.rerank,
        "rerank_top_k": args.rerank_top_k,
        "candidates": args.candidates,
    }

    result = run(cfg)
    print(f"\n{result['n_questions']} questions | {result['config']}")
    print(f"embed={result['embed_model']} chunk_tokens={result['chunk_max_tokens']}\n")
    print(fmt("overall", result["overall"]))
    print("\nby topic:")
    for name, summary in result["by_topic"].items():
        print(fmt(name, summary))
    print("\nby question type:")
    for name, summary in result["by_synthesizer"].items():
        print(fmt(name.replace("_query_synthesizer", ""), summary))

    if not args.no_save:
        stamp = result["run_at"].replace(":", "").replace("-", "")[:15]
        path = results_dir() / f"free-{stamp}Z.json"
        path.write_text(json.dumps(result, indent=2))
        print(f"\n-> {path}")


if __name__ == "__main__":
    main()
