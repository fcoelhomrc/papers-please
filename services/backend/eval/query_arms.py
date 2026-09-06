"""LLM query-side retrieval arms: multi-query, HyDE, and decomposition.

    uv run python -m eval.query_arms generate        # one LLM pass, cached
    uv run python -m eval.query_arms show --arm hyde

What "agentic RAG" meant in this repo until now was a ReAct loop that picked a
query string and decided when to stop. These are the query-side techniques that
were deferred in docs/improvement-plan.md with an explicit trigger - "revisit if
recall on the sweep stalls below ~0.9" - which recall@10 of 0.723, and multi-hop
recall of 0.50-0.62, has comfortably tripped.

Why this is cheap
-----------------
A transformed query is a **pure function of the question**. It does not depend
on top_k, on the retrieval mode, on the reranker, or on anything else the
ablation sweeps. So the LLM runs once per question per arm, the result is
cached to disk, and every subsequent sweep re-reads it for free - which is what
lets these arms live in the *free* branch despite being LLM-driven. Chunk-id
scoring does not care that a model wrote the query.

The three arms
--------------
`multi_query`  3 paraphrases, retrieved separately, fused with RRF. Widens the
               net for one information need.
`hyde`         one hypothetical answer passage, embedded *instead of* the
               question, dense search only. Matches paper-voice against
               paper-voice rather than question-voice against paper-voice.
               Meaningless for a lexical retriever, which is why it is
               dense-only - see `retrieve_for`.
`decompose`    sub-questions for questions spanning more than one lookup,
               retrieved separately and fused. Aimed squarely at the multi-hop
               gap, and reported sliced by synthesizer for that reason: 46 of
               the 100 questions are single-hop and it should not help them.

Fusion, and why it is not free of assumptions
---------------------------------------------
Multi-query and decomposition produce several ranked lists for one question and
fuse them with the same `rrf_fuse` retrieval uses. Every sub-query is weighted
equally, which assumes they are equally good - the same assumption that made
`keyword_weight: 0.1` necessary on the retriever side. Worth revisiting if an
arm underperforms: a bad third paraphrase can outvote two good ones.
"""
import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TESTSET_DIR = Path(__file__).parent / "testset"

MULTI_QUERY = "multi_query"
HYDE = "hyde"
DECOMPOSE = "decompose"
ARMS = (MULTI_QUERY, HYDE, DECOMPOSE)

# HyDE embeds a passage, so it only means something to the dense retriever.
# The others produce ordinary queries and work in any mode.
DENSE_ONLY = (HYDE,)


def cache_path(arm: str) -> Path:
    return TESTSET_DIR / f"queries-{arm}.json"


def cache_key(model: str, version: str) -> str:
    """What invalidates a cached transform.

    Model and prompt version both change the queries, so both belong in the
    key - a prompt edit that silently reused yesterday's queries would report
    the new prompt's score for the old prompt's output.
    """
    return f"{model}@{version}"


def parse_lines(text: str, limit: int) -> list[str]:
    """Non-empty lines, stripped of numbering the model may add anyway."""
    import re

    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if line:
            out.append(line)
    return out[:limit]


def parse(arm: str, text: str) -> list[str]:
    if arm == HYDE:
        # One passage, kept whole - newlines inside it are paragraph breaks,
        # not separate queries.
        return [text.strip()]
    return parse_lines(text, limit=3)


async def transform_one(arm: str, question: str, llm, prompt: str) -> list[str]:
    from langchain_core.messages import HumanMessage, SystemMessage

    reply = await llm.ainvoke(
        [SystemMessage(content=prompt), HumanMessage(content=question)]
    )
    parsed = parse(arm, reply.content)
    # A transform that returns nothing must fall back to the original question
    # rather than retrieving on an empty string, which scores 0 and looks like
    # a retrieval failure instead of a generation one.
    return parsed or [question]


async def generate(arm: str, rows: list[dict], llm, model: str, version: str) -> dict:
    """Transform every question under one arm. Returns {qid: [queries]}."""
    import asyncio

    from prompts.registry import load_prompt

    prompt = load_prompt(arm, version)
    sem = asyncio.Semaphore(8)

    async def one(row):
        async with sem:
            return row["id"], await transform_one(arm, row["question"], llm, prompt)

    pairs = await asyncio.gather(*(one(r) for r in rows))
    return dict(pairs)


def load_cached(arm: str, model: str, version: str) -> dict[str, list[str]] | None:
    path = cache_path(arm)
    if not path.is_file():
        return None
    blob = json.loads(path.read_text())
    if blob.get("key") != cache_key(model, version):
        logger.info(f"{arm}: cache key changed ({blob.get('key')} -> {cache_key(model, version)})")
        return None
    return blob["queries"]


def save_cached(arm: str, queries: dict, model: str, version: str) -> Path:
    path = cache_path(arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"arm": arm, "key": cache_key(model, version), "queries": queries}, indent=2
        )
    )
    return path


def retrieve_for(engine, arm: str, queries: list[str], top_k: int, cfg, mode: str) -> list[dict]:
    """Ranked chunks for one question under one arm.

    HyDE is forced to the dense path regardless of `mode`: the transform's whole
    premise is that the *embedding* of a paper-shaped passage sits nearer the
    real passage than the embedding of a question does. Handing that passage to
    a lexical retriever just searches for the words the model happened to
    invent, which is a different and much worse technique.
    """
    from search import SEMANTIC, rrf_fuse

    if arm in DENSE_ONLY:
        return engine._vector_candidates(queries[0], top_k)

    if len(queries) == 1:
        return _single(engine, queries[0], top_k, mode)

    # One ranked list per sub-query, fused. Each list is pulled at top_k rather
    # than top_k/n: fusion should choose among full lists, not among pre-cut
    # ones, or a chunk ranked 8th by every sub-query is lost before fusion
    # can notice the agreement.
    lists = [_single(engine, q, top_k, mode) for q in queries]
    return rrf_fuse(lists, k=cfg.rrf_k)[:top_k]


def _single(engine, query: str, top_k: int, mode: str) -> list[dict]:
    from search import BM25, KEYWORD, SEMANTIC

    if mode == SEMANTIC:
        return engine._vector_candidates(query, top_k)
    if mode == BM25:
        return engine._bm25_candidates(query, top_k)
    if mode == KEYWORD:
        return engine._keyword_candidates(query, top_k)
    raise ValueError(f"query arms support single-source modes only, got {mode!r}")


def arm_llm():
    """The generator model, same one that wrote the questions.

    Not the judge and not the pipeline answerer - a query rewriter is closer to
    a test-set author than to either, and reusing the generator keeps the
    three-distinct-families rule intact.
    """
    from config import load
    from orchestrator.llm import openrouter_chat

    cfg = load()
    model = cfg.llm.generator_model or cfg.llm.model
    chat = openrouter_chat(model, cfg.ragas.generator_max_tokens, cfg)
    if cfg.ragas.reasoning_effort:
        chat.extra_body = {"reasoning": {"effort": cfg.ragas.reasoning_effort}}
    return chat, model


def main():
    import asyncio

    parser = argparse.ArgumentParser(description="LLM query transforms, cached to disk")
    parser.add_argument("command", choices=["generate", "show"])
    parser.add_argument("--arm", choices=ARMS, default=None, help="default: all arms")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from eval.review import load_curated

    rows = load_curated()
    arms = [args.arm] if args.arm else list(ARMS)

    if args.command == "show":
        for arm in arms:
            path = cache_path(arm)
            if not path.is_file():
                print(f"{arm}: not generated")
                continue
            blob = json.loads(path.read_text())
            print(f"\n=== {arm} ({blob['key']}) ===")
            for qid, queries in list(blob["queries"].items())[:3]:
                original = next(r["question"] for r in rows if r["id"] == qid)
                print(f"\n{qid}: {original[:100]}")
                for q in queries:
                    print(f"   -> {q[:160]}")
        return

    chat, model = arm_llm()
    for arm in arms:
        if not args.force and load_cached(arm, model, args.version) is not None:
            print(f"{arm}: cached, skipping (use --force to regenerate)")
            continue
        print(f"{arm}: transforming {len(rows)} questions with {model}")
        queries = asyncio.run(generate(arm, rows, chat, model, args.version))
        path = save_cached(arm, queries, model, args.version)
        produced = sum(len(v) for v in queries.values())
        print(f"{arm}: {produced} queries for {len(queries)} questions -> {path}")


if __name__ == "__main__":
    main()
