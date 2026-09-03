"""The free evaluation branch: retrieval quality, no LLM calls.

    uv run python -m eval.run_host run_free                    # config defaults
    uv run python -m eval.run_host run_free --mode hybrid --top-k 10
    uv run python -m eval.run_host run_free --rerank --rerank-top-k 5

Scores retrieval against `reference_chunk_ids` - exact set arithmetic on
integers, because the generator was fed our own chunks and every reference
context resolved back to a row in `chunks` (see eval/testset.py).

Deliberately not ragas' `NonLLMContextPrecision`/`NonLLMContextRecall`. Those
compare retrieved *strings* to reference strings with a Levenshtein threshold
of 0.5, which is a lossy approximation of the identity we already have, and
which silently collapses the moment chunking changes. Same metric family,
worse instrument.

Every number carries a confidence interval, because the interval is free -
it comes from the spread across questions, not from repeated runs - and
because at n=100 a five-point difference is not detectable. See
docs/rag-evaluation.md.
"""
import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from eval.retrieval import RETRIEVAL_METRICS, score_question

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

# Scored on `text`, never `context`. neighbour_window glues adjacent chunks
# into `context` for the generator's benefit; the chunk that actually matched
# is what retrieval should be judged on.
CHUNK_ID = "chunk_id"


def curated(questions: list[dict]) -> list[dict]:
    """Questions to score: the kept ones, or everything if review hasn't run.

    Falling back rather than failing keeps the branch usable during the
    review pass - the numbers move as questions are kept, which is the point
    of it being free.
    """
    kept = [q for q in questions if q.get("decision") == "keep"]
    if kept:
        return kept
    logger.info("no questions kept yet - scoring the full generated set")
    return questions


def retrieve(engine, question: str, cfg: dict) -> list[int]:
    """Ranked chunk ids for one question."""
    response = engine.search(
        question,
        top_k=cfg["top_k"],
        rerank=cfg["rerank"],
        rerank_top_k=cfg["rerank_top_k"],
        mode=cfg["mode"],
        candidates=cfg.get("candidates"),
        # 0: expansion is for the reader, and a widened passage would make
        # the retrieved unit stop matching the unit questions were seeded
        # from.
        neighbour_window=0,
    )
    return [r.chunk_id for r in response.results]


def confidence_interval(values: list[float], z: float = 1.96) -> dict:
    """Mean with a 95% interval from the spread across questions.

    No repeated runs and no extra calls: this is sampling variance - "would a
    different 100 questions give a different mean" - which is the question a
    reported score needs answered. Judge variance is a separate, one-time
    measurement.
    """
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "ci": 0.0, "n": 0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": mean, "ci": 0.0, "n": 1}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return {"mean": mean, "ci": z * math.sqrt(var / n), "n": n}


def score(rows: list[dict], engine, cfg: dict) -> dict:
    """Retrieve and score every question at one configuration."""
    per_question = []
    for row in rows:
        retrieved = retrieve(engine, row["question"], cfg)
        relevant = set(row["reference_chunk_ids"])
        scored = score_question(retrieved, relevant, cfg["top_k"])
        scored["id"] = row["id"]
        scored["topics"] = row.get("topics", [])
        per_question.append(scored)

    answered = [q for q in per_question if not q["abstention"]]
    metrics = {
        m: confidence_interval([q[m] for q in answered]) for m in RETRIEVAL_METRICS
    }
    return {"config": cfg, "metrics": metrics, "per_question": per_question}


def by_topic(per_question: list[dict], metric: str = "ndcg") -> dict:
    """One metric sliced by topic.

    The generated set is skewed - one topic took 75 of 131 questions - so a
    single mean can be carried by whichever topic dominates. This is how you
    see that happening.
    """
    buckets: dict[str, list[float]] = {}
    for q in per_question:
        if q["abstention"] or metric not in q:
            continue
        for topic in q["topics"] or ["(none)"]:
            buckets.setdefault(topic, []).append(q[metric])
    return {t: confidence_interval(v) for t, v in sorted(buckets.items())}


def fmt(entry: dict) -> str:
    return f"{entry['mean']:.3f} ± {entry['ci']:.3f}"


def main():
    parser = argparse.ArgumentParser(description="Retrieval scoring, no LLM calls")
    parser.add_argument("--mode", default=None, help="semantic | keyword | hybrid")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from config import load
    from eval.review import questions
    from search import get_search_engine

    cfg = {
        "mode": args.mode or load().search.mode,
        "top_k": args.top_k,
        "rerank": args.rerank,
        "rerank_top_k": args.rerank_top_k,
        "candidates": args.candidates,
    }

    rows = curated(questions())
    print(f"Scoring {len(rows)} questions at {cfg}\n")

    result = score(rows, get_search_engine(), cfg)
    for metric, entry in result["metrics"].items():
        print(f"  {metric:<10} {fmt(entry)}   (n={entry['n']})")

    print("\nnDCG by topic:")
    for topic, entry in by_topic(result["per_question"]).items():
        print(f"  {topic:<14} {fmt(entry)}   (n={entry['n']})")

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = RESULTS_DIR / f"free-{stamp}.json"
        path.write_text(json.dumps({"kind": "retrieval_free", **result}, indent=2))
        print(f"\n-> {path}")


if __name__ == "__main__":
    main()
