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
from search import BM25, HYBRID, HYBRID_BM25, KEYWORD, SEMANTIC, rrf_fuse

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

MODES = (SEMANTIC, KEYWORD, BM25, HYBRID, HYBRID_BM25)
TOP_KS = (1, 3, 5, 10, 20, 50)

# Which cached list each hybrid fuses with dense. Mirrors search.py's
# _HYBRID_KEYWORD_SIDE; the two must agree or the sweep measures a
# configuration production cannot produce.
LEXICAL_SIDE = {HYBRID: KEYWORD, HYBRID_BM25: BM25}

# Ablation W - fusion weights. keyword_weight was fitted on the 12-document
# corpus where dense was the stronger ranker, and never revisited; at 0.1 a
# rank-1 keyword hit scores 0.00164 against a rank-40 dense hit's 0.0100, so
# the keyword side cannot outrank the dense side anywhere in the pool. That
# alone would explain hybrid measuring the same as semantic, which is why this
# is now an axis rather than a constant.
KEYWORD_WEIGHTS = (0.1, 0.25, 0.5, 0.75, 1.0)
RRF_KS = (10, 60)
# Three depths rather than one, so "weight w is best" cannot turn out to be an
# artefact of the single k it was measured at.
WEIGHT_TOP_KS = (5, 10, 20)

# How wide an FTS pool BM25 reranks. Swept to confirm the metric plateaus -
# BM25 can only reorder what the first stage retrieved, so a narrow pool
# reports ts_rank's recall wearing BM25's name.
BM25_POOLS = (100, 200, 500)

# Stops at 40 because A sets the ceiling. Extend to 80 only if A shows
# recall@40 still climbing - otherwise a wider pool buys nothing and costs
# cross-encoder time on every search.
RERANK_POOLS = (10, 20, 40)
RERANK_TOP_KS = (1, 3, 5, 10)
# None = no floor, which is what shipped and is why retrieval could never
# answer "nothing here is relevant". In the cross-encoder's own logit units.
SCORE_FLOORS = (None, -10.0, -8.0, -6.0)


def cache_candidates(engine, questions: list[str], max_k: int) -> dict[str, dict]:
    """Every source's ranked candidates per question, unfused. The only I/O.

    Unfused on purpose: caching the fused list would pin the hybrid pool to
    max_k and freeze the fusion parameters, and ablation W exists precisely to
    vary them.
    """
    cached = {}
    for i, q in enumerate(questions, 1):
        cached[q] = {
            SEMANTIC: engine._vector_candidates(q, max_k),
            KEYWORD: engine._keyword_candidates(q, max_k),
            BM25: engine._bm25_candidates(q, max_k),
        }
        if i % 20 == 0 or i == len(questions):
            logger.info(f"  cached {i}/{len(questions)}")
    return cached


def candidates_for(
    sources: dict,
    mode: str,
    top_k: int,
    cfg,
    keyword_weight: float | None = None,
    rrf_k: int | None = None,
) -> list[dict]:
    """Reproduce SearchEngine.search()'s candidate list from cached sources.

    `keyword_weight` and `rrf_k` default to the configured values, so an
    unparameterised call reproduces production exactly; ablation W overrides
    them.
    """
    if mode in (SEMANTIC, KEYWORD, BM25):
        return sources[mode][:top_k]

    lexical = LEXICAL_SIDE[mode]
    pool = max(cfg.hybrid_candidates, top_k)
    return rrf_fuse(
        [sources[SEMANTIC][:pool], sources[lexical][:pool]],
        k=cfg.rrf_k if rrf_k is None else rrf_k,
        weights=[1.0, cfg.keyword_weight if keyword_weight is None else keyword_weight],
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


def ablation_w(rows, cached, cfg) -> list[dict]:
    """Fusion weights: keyword_weight x rrf_k, for both hybrids.

    Free, because `candidates_for` re-fuses from the cached unfused source
    lists - these are parameters of the fusion, not a reason to re-retrieve.

    The incumbent (keyword_weight 0.1, rrf_k 60) is in the grid, so the
    comparison against it is a row rather than a separate baseline run.
    """
    results = []
    for mode in (HYBRID, HYBRID_BM25):
        for weight in KEYWORD_WEIGHTS:
            for rrf_k in RRF_KS:
                for top_k in WEIGHT_TOP_KS:
                    ranked = {
                        r["id"]: [
                            c["chunk_id"]
                            for c in candidates_for(
                                cached[r["question"]],
                                mode,
                                top_k,
                                cfg,
                                keyword_weight=weight,
                                rrf_k=rrf_k,
                            )
                        ]
                        for r in rows
                    }
                    per_q = score_rows(rows, ranked, top_k)
                    results.append(
                        {
                            "config": {
                                "mode": mode,
                                "top_k": top_k,
                                "keyword_weight": weight,
                                "rrf_k": rrf_k,
                                "rerank": False,
                            },
                            "summary": summarise(per_q),
                            "per_question": {q["id"]: q for q in per_q},
                        }
                    )
        logger.info(f"  swept {mode}")
    return results


def ablation_pool(rows, engine, cfg) -> list[dict]:
    """How wide an FTS pool BM25 needs before its ranking stops improving.

    The one ablation here that re-retrieves, because the pool is a property of
    the first stage rather than of anything cached. Cheap - one FTS query per
    question per pool size, no dense retrieval and no cross-encoder.

    Read it as a plateau check, not a tuning knob: if recall is still climbing
    at 500 then BM25 is being capped by ts_rank's recall and the mode is
    measuring the wrong thing.
    """
    results = []
    for pool in BM25_POOLS:
        for top_k in (10, 20):
            ranked = {
                r["id"]: [
                    c["chunk_id"]
                    for c in engine._bm25_candidates(r["question"], top_k, pool=pool)
                ]
                for r in rows
            }
            per_q = score_rows(rows, ranked, top_k)
            results.append(
                {
                    "config": {"mode": BM25, "top_k": top_k, "bm25_pool": pool},
                    "summary": summarise(per_q),
                    "per_question": {q["id"]: q for q in per_q},
                }
            )
        logger.info(f"  bm25 pool={pool}")
    return results


def timed_pass(engine, rows, cfg, sample: int = 20) -> list[dict]:
    """Per-stage latency, measured through the real search path.

    Separate from the quality sweep and not derived from it. The sweep caches
    every question's candidates once and rebuilds sixty configurations in
    memory, which is what makes it free - and which means timing it would
    report the cache's latency rather than retrieval's.

    A sample rather than the full set: latency does not vary with the question
    the way relevance does, and this is the only part of the ablation that
    pays real I/O per configuration.
    """
    questions = [r["question"] for r in rows][:sample]
    results = []

    configs = [{"mode": m, "top_k": k, "rerank": False} for m in MODES for k in TOP_KS]
    configs += [
        {"mode": HYBRID, "top_k": k, "rerank": True, "rerank_candidates": pool}
        for pool in RERANK_POOLS
        for k in RERANK_TOP_KS
    ]

    for i, config in enumerate(configs, 1):
        stages: dict[str, list[float]] = {}
        for q in questions:
            response = engine.search(
                q,
                top_k=config["top_k"],
                rerank=config["rerank"],
                rerank_top_k=config["top_k"],
                mode=config["mode"],
                candidates=config.get("rerank_candidates"),
                neighbour_window=0,
            )
            for stage, ms in (response.timings or {}).items():
                stages.setdefault(stage, []).append(ms)

        summary = {}
        for stage, values in stages.items():
            mean, half = mean_ci(values)
            summary[stage] = round(mean, 2)
            summary[f"{stage}_ci"] = round(half, 2)
        results.append({"config": config, "n": len(questions), "latency_ms": summary})
        if i % 10 == 0 or i == len(configs):
            logger.info(f"  timed {i}/{len(configs)} configs")
    return results


# Ablation C - LLM query arms. `none` is the untransformed question, present
# as a row so the comparison is paired against the same retrieval path rather
# than against a number from a different run.
ARMS = ("none", "multi_query", "decompose", "hyde")
# HyDE embeds a passage, so it only means anything to the dense retriever.
# The rest produce ordinary queries and run on either single-source mode.
ARM_MODES = {
    "none": (SEMANTIC, BM25),
    "multi_query": (SEMANTIC, BM25),
    "decompose": (SEMANTIC, BM25),
    "hyde": (SEMANTIC,),
}
ARM_TOP_KS = (5, 10, 20)


def arm_queries(rows) -> dict[str, dict[str, list[str]]]:
    """The cached transform for each arm, keyed by question id.

    Read straight off disk rather than regenerated: the whole reason these
    arms are free to sweep is that a transformed query is a pure function of
    the question, so the LLM ran once and everything after it re-reads.
    """
    import json

    from eval.query_arms import cache_path

    out = {"none": {r["id"]: [r["question"]] for r in rows}}
    for arm in ARMS:
        if arm == "none":
            continue
        path = cache_path(arm)
        if not path.is_file():
            logger.warning(f"{arm}: no cached queries at {path}, skipping")
            continue
        out[arm] = json.loads(path.read_text())["queries"]
    return out


def cache_arm_candidates(engine, queries: dict, max_k: int) -> dict:
    """One retrieval per unique (mode, query string), reused across every k.

    Sub-queries repeat across arms and depths, and the same trick that makes
    ablation A free applies here: retrieve once at max_k, slice afterwards.
    Without it multi-query alone would issue three retrievals per question per
    depth per mode.
    """
    wanted: dict[str, set[str]] = {SEMANTIC: set(), BM25: set()}
    for arm, per_question in queries.items():
        for mode in ARM_MODES.get(arm, ()):
            for qs in per_question.values():
                wanted[mode].update(qs)

    cached: dict[tuple[str, str], list[dict]] = {}
    total = sum(len(v) for v in wanted.values())
    done = 0
    for mode, strings in wanted.items():
        for q in sorted(strings):
            cached[(mode, q)] = (
                engine._vector_candidates(q, max_k)
                if mode == SEMANTIC
                else engine._bm25_candidates(q, max_k)
            )
            done += 1
            if done % 100 == 0 or done == total:
                logger.info(f"  retrieved {done}/{total} unique queries")
    return cached


def ablation_c(rows, engine, cfg) -> list[dict]:
    """Query arms x base mode x top_k, scored on the same questions."""
    queries = arm_queries(rows)
    max_k = max(ARM_TOP_KS)
    cached = cache_arm_candidates(engine, queries, max(max_k, cfg.hybrid_candidates))

    results = []
    for arm in ARMS:
        if arm not in queries:
            continue
        for mode in ARM_MODES[arm]:
            for top_k in ARM_TOP_KS:
                ranked = {}
                for row in rows:
                    qs = queries[arm].get(row["id"]) or [row["question"]]
                    lists = [cached.get((mode, q), []) for q in qs]
                    if len(lists) == 1:
                        chunks = lists[0][:top_k]
                    else:
                        # Each sub-query contributes its full list, so a chunk
                        # ranked mid-table by every sub-query survives to
                        # fusion instead of being cut before agreement shows.
                        chunks = rrf_fuse(lists, k=cfg.rrf_k)[:top_k]
                    ranked[row["id"]] = [c["chunk_id"] for c in chunks]
                per_q = score_rows(rows, ranked, top_k)
                results.append(
                    {
                        "config": {"arm": arm, "mode": mode, "top_k": top_k, "rerank": False},
                        "summary": summarise(per_q),
                        "per_question": {q["id"]: q for q in per_q},
                    }
                )
            logger.info(f"  {arm} x {mode}")
    return results


def arm_report(rows: list[dict]) -> None:
    """Each arm against the untransformed question, same mode and depth.

    Paired, because both answered the same 100 questions - comparing an arm's
    mean against a baseline from a different row would fold question
    difficulty into the difference.
    """
    base = {(r["config"]["mode"], r["config"]["top_k"]): r
            for r in rows if r["config"]["arm"] == "none"}
    print("\n=== query arms vs the untransformed question (ndcg) ===")
    print(f"{'arm':<13} {'mode':<9} {'k':>3} {'arm':>8} {'plain':>8}   difference")
    for r in rows:
        c = r["config"]
        if c["arm"] == "none":
            continue
        plain = base.get((c["mode"], c["top_k"]))
        if not plain:
            continue
        d = compare(r, plain, "ndcg")
        print(f"{c['arm']:<13} {c['mode']:<9} {c['top_k']:>3} "
              f"{r['summary']['ndcg']:>8.3f} {plain['summary']['ndcg']:>8.3f}   "
              f"{d['diff']:+.4f} +/- {d['ci']:.4f}  "
              f"[{'significant' if d['significant'] else 'tie'}]")


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


def run(which: str, timed: bool = False, sample: int = 20) -> dict:
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
    if which in ("w", "all"):
        logger.info("ablation W: fusion weight x rrf_k")
        out["w"] = ablation_w(rows, cached, app.search)
    if which in ("pool", "all"):
        logger.info("ablation POOL: how wide an FTS pool BM25 needs")
        out["pool"] = ablation_pool(rows, engine, app.search)
    if which in ("c", "all"):
        logger.info("ablation C: LLM query arms")
        out["c"] = ablation_c(rows, engine, app.search)
    if timed:
        logger.info(f"timed pass: real search path, {sample} questions per config")
        out["timings"] = timed_pass(engine, rows, app.search, sample=sample)
    return out


def report(out: dict) -> None:
    titles = (
        ("a", "A - no reranking"),
        ("b", "B - reranking"),
        ("w", "W - fusion weights"),
        ("pool", "POOL - bm25 first-stage width"),
        ("c", "C - LLM query arms"),
    )
    for key, title in titles:
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
                f"map={s['map']:.3f}  mrr={s['mrr']:.3f}  {label}"
            )

    if "w" in out:
        weight_report(out["w"])
    if "pool" in out:
        pool_report(out["pool"])
    if "c" in out:
        arm_report(out["c"])
    if "timings" in out:
        latency_report(out["timings"])
    if "a" in out and "b" in out:
        matched_report(out)


def weight_report(rows: list[dict]) -> None:
    """nDCG against keyword_weight, at the configured rrf_k.

    Printed as a grid rather than a top-10 because the question is the shape
    of the curve - is 0.1 the peak or the edge of the range - and a leaderboard
    hides that.
    """
    print("\n=== fusion weight sweep (rrf_k=60, ndcg) ===")
    modes = sorted({r["config"]["mode"] for r in rows})
    print(f"{'weight':>7} " + "".join(f"{m + ' k=' + str(k):>20}" for m in modes for k in WEIGHT_TOP_KS))
    for weight in KEYWORD_WEIGHTS:
        cells = []
        for mode in modes:
            for top_k in WEIGHT_TOP_KS:
                match = [
                    r
                    for r in rows
                    if r["config"]["mode"] == mode
                    and r["config"]["keyword_weight"] == weight
                    and r["config"]["rrf_k"] == 60
                    and r["config"]["top_k"] == top_k
                ]
                cells.append(f"{match[0]['summary']['ndcg']:.3f}" if match else "-")
        marker = "  <- shipped" if weight == 0.1 else ""
        print(f"{weight:>7} " + "".join(f"{c:>20}" for c in cells) + marker)


def pool_report(rows: list[dict]) -> None:
    """Whether BM25's ranking has stopped improving with a wider first stage.

    If recall is still climbing at the widest pool, BM25 is capped by ts_rank's
    recall and the bm25 arm is not measuring what it claims to.
    """
    print("\n=== bm25 first-stage pool ===")
    print(f"{'pool':>6} {'k':>4} {'recall':>9} {'ndcg':>9}")
    for r in sorted(rows, key=lambda r: (r["config"]["top_k"], r["config"]["bm25_pool"])):
        c, s = r["config"], r["summary"]
        print(f"{c['bm25_pool']:>6} {c['top_k']:>4} {s['recall']:>9.3f} {s['ndcg']:>9.3f}")


def latency_report(rows: list[dict]) -> None:
    """Mean per-stage milliseconds, and what the stages fail to account for.

    The unattributed column is the check: a large gap means a stage is not
    being timed, and the accuracy-latency figures would then be drawn against
    a number that is not the cost of retrieval.
    """
    print("\n=== latency, real search path (mean ms) ===")
    stages = ("embed", "pinecone", "hydrate", "keyword_sql", "bm25", "fuse", "rerank")
    header = f"{'mode':>12} {'k':>4} {'pool':>5} {'total':>8}"
    print(header + "".join(f"{s:>11}" for s in stages) + f"{'other':>8}")
    for r in rows:
        c, t = r["config"], r["latency_ms"]
        named = sum(t.get(s, 0.0) for s in stages)
        cells = "".join(f"{t.get(s, 0.0):>11.1f}" for s in stages)
        print(
            f"{c['mode']:>12} {c['top_k']:>4} "
            f"{c.get('rerank_candidates', '-'):>5} {t.get('total', 0.0):>8.1f}"
            + cells
            + f"{t.get('total', 0.0) - named:>8.1f}"
        )


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
    parser.add_argument("which", choices=["a", "b", "c", "w", "pool", "all"])
    parser.add_argument(
        "--timed",
        action="store_true",
        help="also measure per-stage latency through the real search path",
    )
    parser.add_argument("--sample", type=int, default=20, help="questions per timed config")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = run(args.which, timed=args.timed, sample=args.sample)
    report(out)

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = out["run_at"].replace(":", "").replace("-", "")[:15]
        path = RESULTS_DIR / f"ablation-{args.which}-{stamp}Z.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"\n-> {path}")


if __name__ == "__main__":
    main()
