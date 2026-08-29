"""Sweeps retrieval parameters and scores each config with eval/retrieval.py.

    uv run python -m eval.sweep
    uv run python -m eval.sweep --modes semantic,hybrid --top-k 5,10,20

NO LLM CALLS. This uses Postgres, Pinecone and the local embedding/reranker
models only - which is the whole point: the Ragas judge costs API tokens per
metric per question, so it can't be swept, while this can.

Retrieval is cached per (mode, question) at the largest top_k in the sweep
and then sliced, rather than re-queried once per config. For semantic and
keyword that's exact - the top 5 of a top-50 query is the top 5. For hybrid
it's exact as long as the fusion pool is held constant, which it is here.
Reranking is genuinely recomputed per candidate-count, because reranking 10
candidates and reranking 50 do not produce the same top 5 - but only once per
candidate-count, since rerank_top_k just decides where the sorted list is cut.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Document
from eval.retrieval import aggregate, score_question

RESULTS_DIR = Path(__file__).parent / "results"
DATASET_PATH = Path(__file__).parent / "dataset.jsonl"

DEFAULT_MODES = ("semantic", "keyword", "hybrid")
DEFAULT_TOP_KS = (1, 3, 5, 10, 20, 50)
DEFAULT_RERANK_TOP_KS = (1, 3, 5, 10)


def load_labeled_dataset(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    missing = [r["question"] for r in rows if "relevant_source_ids" not in r]
    if missing:
        raise ValueError(
            f"{len(missing)} dataset rows lack relevant_source_ids - retrieval "
            f"metrics need labels (first: {missing[0]!r})"
        )
    return rows


def doc_id_to_source_id(engine) -> dict[int, str]:
    """Retrieval returns Postgres doc_ids; labels are fixture source_ids."""
    with Session(engine) as session:
        return dict(session.execute(select(Document.id, Document.source_id)).all())


def retrieve_all(search_engine, questions: list[str], mode: str, max_k: int) -> dict[str, list]:
    """One retrieval per question, at the widest k in the sweep."""
    out = {}
    for i, q in enumerate(questions, 1):
        resp = search_engine.search(q, top_k=max_k, rerank=False, mode=mode)
        out[q] = [
            {
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "text": r.text,
                "score": r.score,
            }
            for r in resp.results
        ]
        print(f"  [{i}/{len(questions)}] {mode}: {q[:55]!r} -> {len(out[q])} chunks")
    return out


def _to_source_ids(chunks: list[dict], id_map: dict[int, str]) -> list[str]:
    return [id_map[c["doc_id"]] for c in chunks if c["doc_id"] in id_map]


def score_config(rows, cached, id_map, k, rerank_fn=None, candidates=None, missing_ok=False) -> dict:
    """Score one configuration across the whole dataset.

    `cached` maps question -> ranked chunks (raw retrieval, or an
    already-reranked list). `missing_ok` allows questions absent from it -
    a question that retrieved nothing has no reranked entry, and that's an
    empty result, not a bug.
    """
    per_question = []
    for row in rows:
        chunks = cached.get(row["question"], []) if missing_ok else cached[row["question"]]
        chunks = chunks[:candidates] if candidates else chunks
        if rerank_fn is not None and chunks:
            chunks = rerank_fn(row["question"], chunks, k)
        retrieved = _to_source_ids(chunks[:k], id_map)
        per_question.append(score_question(retrieved, set(row["relevant_source_ids"]), k))
    return aggregate(per_question)


def run_sweep(search_engine, rows, id_map, modes, top_ks, rerank_top_ks, rerankers=None) -> list[dict]:
    """`rerankers` maps a model id to a loaded Reranker. Defaults to the one
    the engine already holds (config.search.reranker_model); pass more to
    compare cross-encoders, which is the cheapest retrieval lever there is -
    swapping one needs no re-embedding, unlike changing the encoder."""
    max_k = max(top_ks)
    questions = [r["question"] for r in rows]
    results = []
    rerankers = rerankers or {"configured": search_engine._reranker}

    for mode in modes:
        print(f"\nretrieving for mode={mode} at top_k={max_k}...")
        cached = retrieve_all(search_engine, questions, mode, max_k)

        for top_k in top_ks:
            results.append(
                {
                    "mode": mode, "top_k": top_k, "rerank": False,
                    "rerank_top_k": None, "reranker": None, "k": top_k,
                    **score_config(rows, cached, id_map, k=top_k, candidates=top_k),
                }
            )
            print(f"  {mode} top_k={top_k} rerank=off -> recall={results[-1]['recall']:.3f}")

        # Rerank once per (question, top_k), not once per (question, top_k,
        # rerank_top_k): the cross-encoder scores and sorts the whole
        # candidate list, and rerank_top_k only decides where that sorted
        # list is cut. Reranking again per cut repeated the expensive part
        # (~4x the cross-encoder passes) to compute prefixes of a list we
        # already had - and the cross-encoder dominates this sweep's runtime.
        for name, reranker in rerankers.items():
            for top_k in top_ks:
                reranked = {
                    q: reranker.rerank(q, chunks[:top_k], top_k=top_k)
                    for q, chunks in cached.items()
                    if chunks
                }
                print(f"  {mode} {name} top_k={top_k}: reranked {len(reranked)} question(s)")

                for rtk in rerank_top_ks:
                    if rtk > top_k:
                        continue  # can't cut to more results than were retrieved
                    results.append(
                        {
                            "mode": mode, "top_k": top_k, "rerank": True,
                            "rerank_top_k": rtk, "reranker": name, "k": rtk,
                            **score_config(rows, reranked, id_map, k=rtk, missing_ok=True),
                        }
                    )
                    print(
                        f"  {mode} {name} top_k={top_k} rerank->{rtk} "
                        f"-> recall={results[-1]['recall']:.3f}"
                    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Retrieval parameter sweep (no LLM calls)")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--top-k", default=",".join(map(str, DEFAULT_TOP_KS)))
    parser.add_argument("--rerank-top-k", default=",".join(map(str, DEFAULT_RERANK_TOP_KS)))
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--skip-seed", action="store_true")
    parser.add_argument(
        "--rerankers",
        default="",
        help="comma-separated cross-encoder model ids to compare, e.g. "
        "'cross-encoder/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-base'. "
        "Default: the one in config. Swapping a reranker needs no re-embedding, "
        "which makes this the cheapest quality lever to test.",
    )
    args = parser.parse_args()

    modes = args.modes.split(",")
    top_ks = sorted(int(x) for x in args.top_k.split(","))
    rerank_top_ks = sorted(int(x) for x in args.rerank_top_k.split(","))

    rows = load_labeled_dataset(Path(args.dataset))

    if not args.skip_seed:
        from eval.seed import ensure_fixtures_seeded

        ensure_fixtures_seeded()

    from config import load
    from search import get_search_engine

    search_engine = get_search_engine()
    id_map = doc_id_to_source_id(search_engine.engine)

    rerankers = None
    if args.rerankers:
        from process.embedder import Reranker

        rerankers = {}
        for model_id in args.rerankers.split(","):
            print(f"loading reranker {model_id}...")
            rerankers[model_id] = Reranker(
                model_id=model_id, device=load().devices.reranker
            )

    results = run_sweep(
        search_engine, rows, id_map, modes, top_ks, rerank_top_ks, rerankers
    )

    cfg = load()
    output = {
        "kind": "retrieval_sweep",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "embed_model": cfg.embedder.model,
        "reranker_model": args.rerankers or cfg.search.reranker_model,
        "rrf_k": cfg.search.rrf_k,
        "hybrid_candidates": cfg.search.hybrid_candidates,
        "n_questions": len(rows),
        "results": results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"sweep-{stamp}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {out_path}  ({len(results)} configs)")
    return out_path


if __name__ == "__main__":
    main()
