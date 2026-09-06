"""Retrieval ablations. No LLM calls, so sweep as often as you like.

    uv run python -m eval.ablations a      # mode x top_k, no reranking
    uv run python -m eval.ablations b      # reranking: pool x output x floor
    uv run python -m eval.ablations all

Cost is one run, not sixty-six. Each question's candidates are retrieved once
at `max_k` and every configuration is then reconstructed in memory, so 66
configs issue the same ~100 Pinecone queries a single run does.

**Candidates are cached unfused.** Caching the fused list would pin the hybrid
pool to `max_k`, because RRF depends on how many candidates each source
contributed while `search()` uses `max(hybrid_candidates, top_k)`. A sweep
that fused at pool=50 and sliced to 5 once measured hybrid at nDCG 0.804 where
production scored 0.814 - small, but in the pessimistic direction and entirely
an artefact.

Ablation A - retrieval, no reranking
------------------------------------
mode x top_k, with chunk size and embedding model pinned. Establishes the
ceiling: reranking cannot recover a chunk the first stage never fetched, so
A's recall at a given pool size bounds what B can do with that pool.

Ablation B - reranking
----------------------
`search.py` computes `retrieve_k = max(candidates, top_k)`, so `top_k` and
`rerank_candidates` are not independent - (top_k=50, candidates=20) and
(top_k=50, candidates=50) are the same configuration. Sweeping both produces
duplicate rows and an uninterpretable grid. So B pins `top_k` to the pool and
sweeps the pool directly, with the score floor in the same grid rather than a
separate script: the floor *is* the abstention mechanism, and separating them
is why every historical sweep row read abstention_precision 0.000.

The baseline for B is A at matched *output* size, so the comparison is "at the
same number of returned chunks, does reranking a pool of N beat plain
retrieval?" - which is the question, and is a paired test over the same
questions.
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from eval.retrieval import RETRIEVAL_METRICS, score_question
from eval.run_free import mean_ci, paired_diff_ci, summarise
from search import HYBRID, KEYWORD, SEMANTIC, rrf_fuse

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

MODES = (SEMANTIC, KEYWORD, HYBRID)
TOP_KS = (1, 3, 5, 10, 20, 50)

# Stops at 40 because A sets the ceiling. Extend to 80 only if A shows
# recall@40 still climbing - otherwise a wider pool buys nothing and costs
# cross-encoder time on every search.
RERANK_POOLS = (10, 20, 40)
RERANK_TOP_KS = (1, 3, 5, 10)
# None = no floor, which is what shipped and is why retrieval could never
# answer "nothing here is relevant". In the cross-encoder's own logit units.
SCORE_FLOORS = (None, -10.0, -8.0, -6.0)


def cache_candidates(engine, questions: list[str], max_k: int) -> dict[str, dict]:
    """Both sources' ranked candidates per question, unfused. The only I/O."""
    cached = {}
    for i, q in enumerate(questions, 1):
        cached[q] = {
            "vector": engine._vector_candidates(q, max_k),
            "keyword": engine._keyword_candidates(q, max_k),
        }
        if i % 20 == 0 or i == len(questions):
            logger.info(f"  cached {i}/{len(questions)}")
    return cached


def candidates_for(sources: dict, mode: str, top_k: int, cfg) -> list[dict]:
    """Reproduce SearchEngine.search()'s candidate list from cached sources."""
    if mode == SEMANTIC:
        return sources["vector"][:top_k]
    if mode == KEYWORD:
        return sources["keyword"][:top_k]
    pool = max(cfg.hybrid_candidates, top_k)
    return rrf_fuse(
        [sources["vector"][:pool], sources["keyword"][:pool]],
        k=cfg.rrf_k,
        weights=[1.0, cfg.keyword_weight],
    )[:top_k]


def score_rows(rows: list[dict], ranked: dict[str, list[int]], k: int) -> list[dict]:
    """Per-question scores for one configuration."""
    scored = []
    for row in rows:
        s = score_question(ranked[row["id"]], set(row["reference_chunk_ids"]), k)
        s["id"] = row["id"]
        s["synthesizer"] = row["synthesizer"]
        scored.append(s)
    return scored


def ablation_a(rows, cached, cfg) -> list[dict]:
    """mode x top_k, no reranking."""
    results = []
    for mode in MODES:
        for top_k in TOP_KS:
            ranked = {
                r["id"]: [
                    c["chunk_id"] for c in candidates_for(cached[r["question"]], mode, top_k, cfg)
                ]
                for r in rows
            }
            per_q = score_rows(rows, ranked, top_k)
            results.append(
                {
                    "config": {"mode": mode, "top_k": top_k, "rerank": False},
                    "summary": summarise(per_q),
                    "per_question": {q["id"]: q for q in per_q},
                }
            )
            logger.info(
                f"  {mode:<9} k={top_k:<3} recall={results[-1]['summary']['recall']:.3f}"
            )
    return results


def ablation_b(rows, cached, cfg, reranker) -> list[dict]:
    """Reranking: pool x returned x score floor, at the best mode from A.

    The cross-encoder runs once per (question, pool) and every `rerank_top_k`
    and floor is a slice of that one sorted list - which is what keeps 48
    configurations the cost of 3.
    """
    results = []
    for pool in RERANK_POOLS:
        # top_k pinned to the pool: search() takes max(candidates, top_k), so
        # varying them independently produces duplicate configurations.
        reranked = {}
        for row in rows:
            pooled = candidates_for(cached[row["question"]], HYBRID, pool, cfg)
            reranked[row["id"]] = (
                reranker.rerank(row["question"], pooled) if pooled else []
            )
        logger.info(f"  reranked pool={pool}")

        for top_k in RERANK_TOP_KS:
            for floor in SCORE_FLOORS:
                ranked = {}
                for rid, chunks in reranked.items():
                    kept = chunks[:top_k]
                    if floor is not None:
                        kept = [c for c in kept if c["score"] >= floor]
                    ranked[rid] = [c["chunk_id"] for c in kept]
                per_q = score_rows(rows, ranked, top_k)
                results.append(
                    {
                        "config": {
                            "mode": HYBRID,
                            "rerank": True,
                            "rerank_candidates": pool,
                            "top_k": top_k,
                            "min_rerank_score": floor,
                        },
                        "summary": summarise(per_q),
                        "per_question": {q["id"]: q for q in per_q},
                        "mean_returned": round(
                            sum(len(v) for v in ranked.values()) / len(ranked), 2
                        ),
                    }
                )
    return results


def compare(a: dict, b: dict, metric: str = "recall") -> dict:
    """Paired difference between two configurations on the same questions.

    Paired because both answered the same set, so question difficulty
    cancels. Two overlapping individual intervals do not imply the
    difference is indistinguishable from zero, which is why this exists
    rather than eyeballing the two means.
    """
    ids = [i for i in a["per_question"] if not a["per_question"][i]["abstention"]]
    diff, half = paired_diff_ci(
        [a["per_question"][i][metric] for i in ids],
        [b["per_question"][i][metric] for i in ids],
    )
    return {
        "metric": metric,
        "diff": round(diff, 4),
        "ci": round(half, 4),
        # The interval excluding zero is the claim; anything else is a tie.
        "significant": abs(diff) > half,
        "n": len(ids),
    }


def best(results: list[dict], metric: str = "ndcg") -> dict:
    return max(results, key=lambda r: r["summary"][metric])


def run(which: str) -> dict:
    from config import load
    from eval.review import load_curated
    from process.embedder import Reranker
    from search import get_search_engine

    app = load()
    rows = load_curated()
    engine = get_search_engine()

    max_k = max(max(TOP_KS), max(RERANK_POOLS), app.search.hybrid_candidates)
    logger.info(f"caching candidates for {len(rows)} questions at k={max_k}")
    cached = cache_candidates(engine, [r["question"] for r in rows], max_k)

    out: dict = {
        "kind": "retrieval_ablation",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(rows),
        "embed_model": app.embedder.model,
        "chunk_max_tokens": app.embedder.max_tokens,
        "reranker_model": app.search.reranker_model,
        "corpus": app.search.corpus,
    }

    if which in ("a", "all"):
        logger.info("ablation A: mode x top_k, no reranking")
        out["a"] = ablation_a(rows, cached, app.search)
    if which in ("b", "all"):
        logger.info("ablation B: reranking pool x returned x floor")
        out["b"] = ablation_b(rows, cached, app.search, Reranker(device=app.devices.reranker))
    return out


def report(out: dict) -> None:
    for key, title in (("a", "A - no reranking"), ("b", "B - reranking")):
        if key not in out:
            continue
        print(f"\n=== ablation {title} ===")
        rows = sorted(out[key], key=lambda r: -r["summary"]["ndcg"])
        for r in rows[:10]:
            s, c = r["summary"], r["config"]
            label = ", ".join(f"{k}={v}" for k, v in c.items() if k != "rerank")
            print(
                f"  ndcg={s['ndcg']:.3f}+/-{s['ndcg_ci']:.3f}  "
                f"recall={s['recall']:.3f}+/-{s['recall_ci']:.3f}  "
                f"mrr={s['mrr']:.3f}  {label}"
            )

    if "a" in out and "b" in out:
        matched_report(out)


def matched_report(out: dict) -> None:
    """Reranked vs plain at the *same number of returned chunks*.

    Comparing the global best of each ablation is meaningless: A's winner
    returns 50 chunks and B's returns 10, and recall rises with k
    mechanically, so the naive comparison reported reranking as
    significantly worse when it was simply returning a fifth as much. The
    only fair question is "at the same output size, does reranking a pool
    beat plain retrieval?"
    """
    by_mode = {(r["config"]["mode"], r["config"]["top_k"]): r for r in out["a"]}
    print("\n=== reranked vs plain, matched output size ===")
    print(f"{'k':>4} {'pool':>5} {'rerank':>8} {'hybrid':>8} {'keyword':>8}   rerank - hybrid (ndcg)")
    for k in sorted({r["config"]["top_k"] for r in out["b"]}):
        at_k = [r for r in out["b"] if r["config"]["top_k"] == k]
        if not at_k or ("hybrid", k) not in by_mode:
            continue
        top = max(at_k, key=lambda r: r["summary"]["ndcg"])
        plain = by_mode[("hybrid", k)]
        d = compare(top, plain, "ndcg")
        keyword = by_mode.get(("keyword", k))
        print(
            f"{k:>4} {top['config']['rerank_candidates']:>5} "
            f"{top['summary']['ndcg']:>8.3f} {plain['summary']['ndcg']:>8.3f} "
            f"{keyword['summary']['ndcg'] if keyword else float('nan'):>8.3f}   "
            f"{d['diff']:+.4f} +/- {d['ci']:.4f}  "
            f"[{'significant' if d['significant'] else 'tie'}, n={d['n']}]"
        )


def main():
    parser = argparse.ArgumentParser(description="Retrieval ablations, no LLM calls")
    parser.add_argument("which", choices=["a", "b", "all"])
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = run(args.which)
    report(out)

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = out["run_at"].replace(":", "").replace("-", "")[:15]
        path = RESULTS_DIR / f"ablation-{args.which}-{stamp}Z.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"\n-> {path}")


if __name__ == "__main__":
    main()
