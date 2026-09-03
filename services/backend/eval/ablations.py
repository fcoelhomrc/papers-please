"""Retrieval ablations. No LLM calls, so run them as often as you like.

    uv run python -m eval.run_host ablations a     # mode x top_k, no rerank
    uv run python -m eval.run_host ablations b     # reranking grid

**A - retrieval.** mode x top_k with reranking off, at fixed chunk size and
embedding model. 18 configs.

**B - reranking.** `search()` computes `retrieve_k = max(candidates, top_k)`,
so `top_k` and `rerank_candidates` are *not* independent: (top_k=50,
candidates=20) and (top_k=50, candidates=50) are the same config and would
appear twice in a grid that swept both. B pins top_k to the pool and sweeps
the pool directly, then compares against A at matched *output* size - "at the
same number of returned chunks, does reranking a pool of N beat plain
retrieval?"

The pool ladder stops at 40 because A determines the ceiling: reranking cannot
recover a chunk the first stage never retrieved. Extend to 80 only if A shows
recall@40 still climbing.

Candidates are retrieved once per question at the widest k and re-scored per
configuration, rather than re-querying for every row.
"""
import argparse
import json
import logging
from datetime import datetime, timezone

from eval.retrieval import RETRIEVAL_METRICS, score_question
from eval.run_free import RESULTS_DIR, confidence_interval, curated, fmt

logger = logging.getLogger(__name__)

MODES = ("semantic", "keyword", "hybrid")
TOP_KS = (1, 3, 5, 10, 20, 50)

# Pool sizes fed to the cross-encoder; top_k is pinned to each.
POOLS = (10, 20, 40)
RERANK_TOP_KS = (1, 3, 5, 10)
# None = no floor. The floor is the abstention mechanism, so it belongs in
# this grid rather than a separate script - separating them is why every
# historical sweep reported abstention_precision 0.000.
FLOORS = (None, -10.0, -8.0, -6.0)


def grid_a() -> list[dict]:
    return [
        {"mode": m, "top_k": k, "rerank": False, "rerank_top_k": k, "candidates": None}
        for m in MODES
        for k in TOP_KS
    ]


def grid_b(mode: str) -> list[dict]:
    return [
        {
            "mode": mode,
            "top_k": pool,
            "rerank": True,
            "rerank_top_k": out,
            "candidates": pool,
            "min_rerank_score": floor,
        }
        for pool in POOLS
        for out in RERANK_TOP_KS
        for floor in FLOORS
        if out <= pool
    ]


def run_config(engine, rows: list[dict], cfg: dict) -> dict:
    per_question = []
    for row in rows:
        thresholds = (
            {"min_rerank_score": cfg["min_rerank_score"]}
            if "min_rerank_score" in cfg
            else None
        )
        response = engine.search(
            row["question"],
            top_k=cfg["top_k"],
            rerank=cfg["rerank"],
            rerank_top_k=cfg["rerank_top_k"],
            mode=cfg["mode"],
            candidates=cfg.get("candidates"),
            thresholds=thresholds,
            neighbour_window=0,
        )
        retrieved = [r.chunk_id for r in response.results]
        # Judged at the number actually returned, not the retrieval depth -
        # with reranking on, rerank_top_k is the output size.
        k = cfg["rerank_top_k"] if cfg["rerank"] else cfg["top_k"]
        scored = score_question(retrieved, set(row["reference_chunk_ids"]), k)
        scored["id"] = row["id"]
        per_question.append(scored)

    answered = [q for q in per_question if not q["abstention"]]
    return {
        **cfg,
        **{
            m: confidence_interval([q[m] for q in answered])["mean"]
            for m in RETRIEVAL_METRICS
        },
        "ci_ndcg": confidence_interval([q["ndcg"] for q in answered])["ci"],
        "n": len(answered),
    }


def pareto(results: list[dict], x: str = "recall", y: str = "precision") -> list[dict]:
    """Configs no other config beats on both axes.

    Selecting on one metric alone always picks its extreme - argmax-recall is
    always the widest k - so the frontier is what a choice should be made
    from.
    """
    frontier = []
    for r in results:
        if not any(o[x] >= r[x] and o[y] >= r[y] and o is not r for o in results):
            frontier.append(r)
    return sorted(frontier, key=lambda r: r[x])


def main():
    parser = argparse.ArgumentParser(description="Retrieval ablations, no LLM calls")
    parser.add_argument("ablation", choices=["a", "b"])
    parser.add_argument("--mode", default="hybrid", help="ablation B only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from eval.review import questions
    from search import get_search_engine

    rows = curated(questions())
    configs = grid_a() if args.ablation == "a" else grid_b(args.mode)
    engine = get_search_engine()
    print(f"{len(configs)} configs x {len(rows)} questions\n")

    results = []
    for i, cfg in enumerate(configs, 1):
        results.append(run_config(engine, rows, cfg))
        print(f"  [{i}/{len(configs)}] {cfg}", flush=True)

    results.sort(key=lambda r: r["ndcg"], reverse=True)
    print(f"\nTop 5 by nDCG:")
    for r in results[:5]:
        label = f"{r['mode']} k={r['top_k']}" + (
            f" rerank->{r['rerank_top_k']} floor={r.get('min_rerank_score')}"
            if r["rerank"]
            else ""
        )
        print(f"  {label:<48} ndcg {r['ndcg']:.3f} ± {r['ci_ndcg']:.3f}  recall {r['recall']:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"ablation-{args.ablation}-{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "kind": f"ablation_{args.ablation}",
                "n_questions": len(rows),
                "results": results,
                "pareto": pareto(results),
            },
            indent=2,
        )
    )
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
