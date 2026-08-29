"""Finds the score floor that lets retrieval abstain without losing recall.

    uv run python -m eval.thresholds

NO LLM CALLS. Retrieval is done once per (mode, question) with no floor, then
every candidate floor is applied to those cached results locally - filtering a
cached list by score is exact, so sweeping 20 thresholds costs one retrieval
pass, not 20.

The trade-off being measured: a floor raises `abstention_precision` (retrieval
correctly returns nothing on the 8 questions where the library has nothing
relevant) and lowers `recall` (a real hit scoring below the floor is dropped).
The useful floor is where the first is already paid for and the second has not
started - a knee to read off the curve, not a number to guess.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from eval.retrieval import aggregate, score_question
from eval.sweep import (
    DATASET_PATH,
    RESULTS_DIR,
    _to_source_ids,
    doc_id_to_source_id,
    load_labeled_dataset,
)
from search import HYBRID, KEYWORD, SEMANTIC, rrf_fuse

# cosine over normalised bge embeddings: in practice everything lands in
# roughly 0.35-0.90, so this brackets the whole usable band.
DEFAULT_VECTOR_FLOORS = (0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


def cache_sources(engine, questions, max_k):
    """Retrieve both sources once per question, unfiltered."""
    out = {}
    for i, q in enumerate(questions, 1):
        out[q] = {
            "vector": engine._vector_candidates(q, max_k),
            "keyword": engine._keyword_candidates(q, max_k),
        }
        print(f"  [{i}/{len(questions)}] cached {len(out[q]['vector'])}v/{len(out[q]['keyword'])}k")
    return out


def candidates_for(sources, mode, floor, rrf_k, top_k, kw_floor=0.0, kw_weight=1.0):
    """Apply a vector floor, then build the mode's ranked list from cache.

    The floor is applied per source before fusion, matching what
    SearchEngine.search does - an RRF score is a rank artefact with no notion
    of 'relevant enough', so it cannot be filtered after the fact.
    """
    vec = [c for c in sources["vector"] if c["score"] >= floor]
    kw = [c for c in sources["keyword"] if c["score"] >= kw_floor]
    if mode == SEMANTIC:
        return vec[:top_k]
    if mode == KEYWORD:
        return kw[:top_k]
    return rrf_fuse([vec, kw], k=rrf_k, weights=[1.0, kw_weight])[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Score-floor sweep (no LLM calls)")
    parser.add_argument("--modes", default=f"{SEMANTIC},{HYBRID}")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--floors", default=",".join(str(f) for f in DEFAULT_VECTOR_FLOORS))
    parser.add_argument(
        "--kw-floors",
        default="0.0",
        help="ts_rank floors for the keyword source. Needed once keyword "
        "matching is OR-based: it then matches hundreds of weakly-related "
        "chunks, and RRF gives every one of them rank credit.",
    )
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    args = parser.parse_args()

    rows = load_labeled_dataset(Path(args.dataset))
    floors = [float(f) for f in args.floors.split(",")]
    kw_floors = [float(f) for f in args.kw_floors.split(",")]

    from config import load
    from search import get_search_engine

    engine = get_search_engine()
    id_map = doc_id_to_source_id(engine.engine)
    cfg = load()

    print(f"caching retrieval for {len(rows)} questions...")
    sources = cache_sources(engine, [r["question"] for r in rows], max_k=50)

    results = []
    for mode in args.modes.split(","):
        for floor in floors:
            for kwf in kw_floors:
                per_q = []
                for row in rows:
                    chunks = candidates_for(
                        sources[row["question"]], mode, floor, cfg.search.rrf_k,
                        args.top_k, kw_floor=kwf, kw_weight=cfg.search.keyword_weight,
                    )
                    retrieved = _to_source_ids(chunks, id_map)
                    per_q.append(
                        score_question(retrieved, set(row["relevant_source_ids"]), args.top_k)
                    )
                agg = aggregate(per_q)
                results.append({
                    "mode": mode, "min_vector_score": floor,
                    "min_keyword_score": kwf, "top_k": args.top_k, **agg,
                })
                print(
                    f"  {mode} vec>={floor:.2f} kw>={kwf:.3f} recall={agg['recall']:.3f} "
                    f"ndcg={agg['ndcg']:.3f} abstention={agg.get('abstention_precision', 0):.3f} "
                    f"fp={agg.get('mean_false_positives', 0):.2f}"
                )

    out = {
        "kind": "threshold_sweep",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "embed_model": cfg.embedder.model,
        "top_k": args.top_k,
        "n_questions": len(rows),
        "results": results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"thresholds-{stamp}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    return path


if __name__ == "__main__":
    main()
