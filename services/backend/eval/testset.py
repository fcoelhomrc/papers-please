"""Generate the evaluation question set with ragas.

    uv run python -m eval.testset seed        # sample the seed pool
    uv run python -m eval.testset kg          # build + save the knowledge graph
    uv run python -m eval.testset generate    # write generated.jsonl

Three stages, each persisted, because the knowledge graph is the expensive
part (four LLM calls per node) and regenerating questions from a saved graph
costs two calls each.

Two ragas behaviours this exists to work around
-----------------------------------------------
**`default_transforms` would re-split our chunks.** It branches on input
length and inserts a `HeadlineSplitter` when >=25% of documents exceed 500
tokens. Our chunks are already the retrieval unit; re-splitting them breaks
the correspondence between a graph node and a row in `chunks`, and that
correspondence is the whole point (see `map_chunk_ids`). So the transform list
is built here rather than inferred, and no splitter is ever in it.

**`default_query_distribution` drops synthesizers in silence.** It calls
`get_node_clusters` on each one and discards any that comes back empty, with
no warning. Multi-hop specific needs `entities_overlap` relationships and
multi-hop abstract needs `summary_similarity` ones; on a topically scattered
corpus neither exists, every multi-hop synthesizer disappears, and the result
is an all-single-hop test set that looks perfectly healthy. `assert_clusters`
turns that into a loud failure.

Why the seed pool is a sample
-----------------------------
Questions are seeded from ~400 chunks while retrieval searches all ~7,000.
Generation cost scales with the pool; retrieval difficulty scales with the
haystack. Sampling the first and not the second is what makes this both cheap
and hard.
"""
import argparse
import json
import logging
import random
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
TESTSET_DIR = HERE / "testset"
SEED_PATH = TESTSET_DIR / "seed_pool.jsonl"
PERSONAS_PATH = TESTSET_DIR / "personas.json"
KG_PATH = TESTSET_DIR / "kg.json"
GENERATED_PATH = TESTSET_DIR / "generated.jsonl"

SEED_PER_TOPIC = 80
TESTSET_SIZE = 130

# SummaryExtractor's own gate is 100 tokens; a chunk below it gets no summary,
# so it can never join a summary_similarity cluster and is dead weight in the
# graph. Filtering here rather than letting the extractor skip it keeps the
# pool size honest.
MIN_CHUNK_TOKENS = 100

# Deterministic pool: re-running the seed stage must not silently change what
# the questions were drawn from, or two runs of the same experiment are
# measuring different things.
SEED = 0

# Cosine similarity above which two chunk summaries count as related.
#
# NOT ragas' 0.9 default, and not the 0.5 its short-document branch uses -
# both assume a topically diverse corpus. Ours is 100 ML papers, so every
# summary resembles every other: measured over this corpus the *minimum*
# pairwise similarity is 0.500 and the median 0.647, so 0.5 admitted 75,223
# of 79,401 possible pairs (94.7%) - a complete graph, not a similarity
# graph. `find_indirect_clusters` at depth 3 over that never returns; it hung
# a run for 13 minutes at 0% CPU before it was killed.
#
# Measured on this corpus:
#   0.85 ->    126 rels ->   394 clusters in  0.1s
#   0.80 ->    788 rels -> 8,341 clusters in 12.9s
#   0.75 ->  4,047 rels -> timed out at 60s
#   0.70 -> 15,490 rels -> timed out at 60s
#
# 0.80 gives far more clusters than the ~32 multi-hop-abstract questions
# need, and stays tractable.
COSINE_THRESHOLD = 0.80

# 50/25/25 rather than ragas' default third each. Single-hop questions are the
# ones a retrieval ablation can actually discriminate on; multi-hop are the
# ones that catch a pipeline that retrieves one good chunk and stops.
DISTRIBUTION = {
    "single_hop_specific": 0.50,
    "multi_hop_specific": 0.25,
    "multi_hop_abstract": 0.25,
}


# --- Stage 1: seed pool ----------------------------------------------------


def seed_pool(per_topic: int = SEED_PER_TOPIC, seed: int = SEED) -> list[dict]:
    """A stratified sample of eval-corpus chunks, `per_topic` from each topic.

    Stratified rather than random over the whole corpus because chunk counts
    differ several-fold between papers - a 30-page survey would otherwise
    crowd out an entire topic of short papers, and the question set would
    inherit that skew.
    """
    from db.connection import PostgresInterface
    from db.models import Chunk, Document, Object

    stmt = (
        select(Chunk.id, Chunk.chunk_text, Document.id.label("doc_id"), Document.topic)
        .join(Object, Chunk.obj_id == Object.id)
        .join(Document, Object.doc_id == Document.id)
        .where(Document.corpus == "eval")
        .where(Chunk.chunk_text.is_not(None))
        .order_by(Chunk.id)
    )
    with Session(PostgresInterface.connect()) as session:
        rows = session.execute(stmt).all()

    by_topic: dict[str, list[dict]] = {}
    for r in rows:
        if count_tokens(r.chunk_text) < MIN_CHUNK_TOKENS:
            continue
        by_topic.setdefault(r.topic, []).append(
            {"chunk_id": r.id, "doc_id": r.doc_id, "topic": r.topic, "text": r.chunk_text}
        )

    rng = random.Random(seed)
    pool: list[dict] = []
    for topic in sorted(by_topic):
        chunks = by_topic[topic]
        pool.extend(rng.sample(chunks, min(per_topic, len(chunks))))
        logger.info(f"{topic}: {min(per_topic, len(chunks))} of {len(chunks)} eligible")
    return pool


def write_seed_pool(pool: list[dict], path: Path = SEED_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(c) + "\n" for c in pool))
    return path


def read_seed_pool(path: Path = SEED_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- Stage 2: knowledge graph ----------------------------------------------


class PermissiveTokenizer:
    """tiktoken, but special-token literals are ordinary text.

    `LLMBasedExtractor.split_text_by_token_limit` calls `tokenizer.encode`
    unconditionally on every node, and tiktoken's default raises
    `ValueError: Encountered text corresponding to disallowed special token
    '<|endoftext|>'`. A corpus of LLM papers is exactly the corpus that
    quotes those markers in running prose, so the graph build would die
    partway through - after paying for every node before it.

    Fixing the counter rather than the text is deliberate: stripping the
    marker would make a node's `page_content` differ from `chunks.chunk_text`
    and break the exact chunk-id mapping the free metrics rest on.
    """

    def __init__(self, encoding):
        self._encoding = encoding

    def encode(self, text: str):
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, tokens):
        return self._encoding.decode(tokens)


def permissive_tokenizer():
    import tiktoken

    from ragas.testset.transforms.base import DEFAULT_TOKENIZER

    return PermissiveTokenizer(DEFAULT_TOKENIZER)


def count_tokens(text: str) -> int:
    """Token count that does not raise on special-token literals.

    ragas' own `num_tokens_from_string` does - see PermissiveTokenizer.
    """
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))


def extraction_transforms(llm, embeddings):
    """The per-node LLM work: summary, filter, themes, entities, embedding.

    Never `default_transforms`. It branches on input length and inserts a
    `HeadlineSplitter` when >=25% of documents exceed 500 tokens, which would
    re-split chunks that are already the retrieval unit and break the
    correspondence between a graph node and a row in `chunks`.
    """
    from ragas.testset.transforms import CustomNodeFilter, EmbeddingExtractor, Parallel
    from ragas.testset.transforms.extractors.llm_based import (
        NERExtractor,
        SummaryExtractor,
        ThemesExtractor,
    )

    tokenizer = permissive_tokenizer()
    return [
        SummaryExtractor(llm=llm, tokenizer=tokenizer),
        CustomNodeFilter(llm=llm),
        Parallel(
            EmbeddingExtractor(
                embedding_model=embeddings,
                property_name="summary_embedding",
                embed_property_name="summary",
            ),
            ThemesExtractor(llm=llm, tokenizer=tokenizer),
            NERExtractor(llm=llm, tokenizer=tokenizer),
        ),
    ]


def relationship_transforms():
    """The edges each multi-hop synthesizer draws from. No LLM calls.

      summary_similarity  -> multi-hop abstract
      entities_overlap    -> multi-hop specific

    Kept separate from extraction so incomplete nodes can be pruned in
    between - see `prune_incomplete`.
    """
    from ragas.testset.transforms import (
        CosineSimilarityBuilder,
        OverlapScoreBuilder,
        Parallel,
    )

    return [
        Parallel(
            CosineSimilarityBuilder(
                property_name="summary_embedding",
                new_property_name="summary_similarity",
                threshold=COSINE_THRESHOLD,
            ),
            OverlapScoreBuilder(threshold=0.01),
        )
    ]


def prune_incomplete(kg) -> list:
    """Drop nodes missing a property the relationship builders require.

    One failed extraction must not cost a whole relationship type.
    `OverlapScoreBuilder` raises "Node X or Y has no entities" on the first
    pair involving an incomplete node and aborts the entire builder - which
    is how a run produced 75,223 cosine relationships and *zero*
    entities_overlap ones, silently removing multi-hop-specific questions
    from a test set that still looked complete.
    """
    required = ("entities", "summary_embedding")
    dropped = [
        n for n in kg.nodes if any(n.properties.get(p) is None for p in required)
    ]
    for node in dropped:
        kg.remove_node(node)
    if dropped:
        logger.info(f"pruned {len(dropped)} nodes missing one of {required}")
    return dropped


def build_kg(pool: list[dict], llm, embeddings):
    """One node per chunk, then the transforms. Returns the graph.

    Nodes are DOCUMENT type deliberately. CHUNK is what a splitter produces,
    and every `filter_nodes` predicate in ragas keys off the type - marking
    unsplit chunks as CHUNK would route them through the wrong filters.
    """
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import apply_transforms

    kg = KnowledgeGraph()
    for c in pool:
        kg.add(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": c["text"],
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "topic": c["topic"],
                },
            )
        )
    logger.info(f"Applying transforms to {len(kg.nodes)} nodes")
    apply_transforms(kg, extraction_transforms(llm, embeddings), run_config())
    prune_incomplete(kg)
    apply_transforms(kg, relationship_transforms(), run_config())
    return kg


def cluster_counts(kg) -> dict[str, int]:
    """How many clusters each synthesizer would find. Zero means it is dropped."""
    from ragas.testset.synthesizers.multi_hop.abstract import (
        MultiHopAbstractQuerySynthesizer,
    )
    from ragas.testset.synthesizers.multi_hop.specific import (
        MultiHopSpecificQuerySynthesizer,
    )
    from ragas.testset.synthesizers.single_hop.specific import (
        SingleHopSpecificQuerySynthesizer,
    )

    return {
        "single_hop_specific": len(
            SingleHopSpecificQuerySynthesizer(llm=None).get_node_clusters(kg)
        ),
        "multi_hop_abstract": len(
            MultiHopAbstractQuerySynthesizer(llm=None).get_node_clusters(kg)
        ),
        "multi_hop_specific": len(
            MultiHopSpecificQuerySynthesizer(llm=None).get_node_clusters(kg)
        ),
    }


def assert_clusters(kg) -> dict[str, int]:
    """Fail loudly rather than generate a silently single-hop test set.

    `default_query_distribution` drops a synthesizer whose clusters are empty
    and says nothing about it. A 50/25/25 split that quietly became 100/0/0
    still produces 130 plausible questions and a clean run.
    """
    counts = cluster_counts(kg)
    empty = [name for name, n in counts.items() if n == 0]
    if empty:
        raise RuntimeError(
            f"no clusters for {', '.join(empty)} - ragas would drop "
            f"{'them' if len(empty) > 1 else 'it'} silently. Counts: {counts}. "
            "Usually means the corpus is too topically scattered for the "
            "similarity thresholds in transforms()."
        )
    return counts


# --- Stage 3: generate -----------------------------------------------------


def query_distribution(llm, split: dict[str, float] | None = None):
    """The 50/25/25 split, built explicitly.

    ragas' own `default_query_distribution` weights a third each *and* drops
    synthesizers with no clusters, so it can neither be trusted for the ratio
    nor relied on to complain.
    """
    from ragas.testset.synthesizers.multi_hop.abstract import (
        MultiHopAbstractQuerySynthesizer,
    )
    from ragas.testset.synthesizers.multi_hop.specific import (
        MultiHopSpecificQuerySynthesizer,
    )
    from ragas.testset.synthesizers.single_hop.specific import (
        SingleHopSpecificQuerySynthesizer,
    )

    split = split or DISTRIBUTION
    classes = {
        "single_hop_specific": SingleHopSpecificQuerySynthesizer,
        "multi_hop_specific": MultiHopSpecificQuerySynthesizer,
        "multi_hop_abstract": MultiHopAbstractQuerySynthesizer,
    }
    return [(classes[name](llm=llm), weight) for name, weight in split.items()]


def map_chunk_ids(reference_contexts: list[str], lookup: dict[str, int]) -> list[int]:
    """Resolve each reference context back to the chunk row it came from.

    Exact string equality, not similarity. Because no splitter ran, a node's
    `page_content` is byte-identical to `chunks.chunk_text`, so this is a
    dictionary hit rather than a threshold judgement - which is what lets the
    free metrics be exact set arithmetic on integers instead of Levenshtein
    against a 0.5 cutoff that breaks the moment chunking changes.

    A context that does not resolve is dropped and counted by the caller: it
    means ragas transformed the text somewhere, and a silently empty
    `reference_chunk_ids` would read as "retrieval found nothing" forever.
    """
    return [lookup[c] for c in reference_contexts if c in lookup]


def to_rows(testset, pool: list[dict]) -> tuple[list[dict], int]:
    """Testset -> jsonl rows. Returns (rows, unmapped_context_count)."""
    lookup = {c["text"]: c["chunk_id"] for c in pool}
    doc_of = {c["chunk_id"]: c["doc_id"] for c in pool}
    topic_of = {c["chunk_id"]: c["topic"] for c in pool}

    rows, unmapped = [], 0
    for i, sample in enumerate(testset.samples):
        eval_sample = sample.eval_sample
        contexts = list(eval_sample.reference_contexts or [])
        chunk_ids = map_chunk_ids(contexts, lookup)
        unmapped += len(contexts) - len(chunk_ids)
        rows.append(
            {
                "id": f"q{i:04d}",
                "question": eval_sample.user_input,
                "reference": eval_sample.reference,
                "reference_contexts": contexts,
                "reference_chunk_ids": chunk_ids,
                "reference_doc_ids": sorted({doc_of[c] for c in chunk_ids}),
                "topics": sorted({topic_of[c] for c in chunk_ids}),
                "synthesizer": sample.synthesizer_name,
            }
        )
    return rows, unmapped


# --- Wiring ----------------------------------------------------------------


def run_config():
    """Concurrency and retry limits, from config.ragas.

    Ragas defaults to max_workers=16 / max_retries=10, which against
    OpenRouter is the expensive failure mode: sixteen concurrent requests
    trip the rate limit and ten retries per call keep paying for retries
    rather than stopping and saying so.
    """
    from config import load
    from ragas.run_config import RunConfig

    r = load().ragas
    return RunConfig(
        max_workers=r.max_workers, max_retries=r.max_retries, timeout=r.timeout_s
    )


def generator_llm(cost_handler=None):
    """The model that writes the questions, plus its id.

    Deliberately not the pipeline model and not the judge: a model that wrote
    the exam should not also sit it or mark it. See docs/rag-evaluation.md.

    The cost handler is attached to the *LangChain model*, not passed to
    ragas. `apply_transforms` accepts a `callbacks` argument and never uses
    it, so a handler given to ragas collects nothing from the graph build -
    which is the bulk of the spend. Attached here it sees every call from
    every caller: extraction, persona generation and synthesis alike.
    """
    from config import load
    from orchestrator.llm import openrouter_chat
    from ragas.llms import LangchainLLMWrapper

    cfg = load()
    model = cfg.llm.generator_model or cfg.llm.model
    # 2048 rather than the agent's 512: ragas' extraction prompts return
    # structured JSON and truncate into LLMDidNotFinishException at 512.
    chat = openrouter_chat(model, 2048, cfg)
    if cost_handler is not None:
        chat.callbacks = [cost_handler]
    wrapped = LangchainLLMWrapper(chat)
    wrapped.set_run_config(run_config())
    return wrapped, model


def cost_handler():
    """Counts tokens across every call the generator makes."""
    from config import load
    from ragas.cost import (
        CostCallbackHandler,
        get_token_usage_for_anthropic,
        get_token_usage_for_openai,
    )

    # The wire shapes differ - Anthropic reports usage.input_tokens,
    # OpenAI-compatible endpoints report token_usage.prompt_tokens - and the
    # wrong parser returns zeros silently rather than failing.
    parser = (
        get_token_usage_for_anthropic
        if load().llm.provider == "anthropic"
        else get_token_usage_for_openai
    )
    return CostCallbackHandler(token_usage_parser=parser)


def report_spend(handler, model: str, label: str) -> None:
    """Print tokens and, where the rate is known, dollars.

    An unlisted model reports tokens with no dollar figure rather than
    inventing one - see eval/pricing.py.
    """
    from eval.pricing import model_price

    if not handler.usage_data:
        print(f"  {label}: no usage recorded")
        return

    usage = handler.total_tokens()
    usages = usage if isinstance(usage, list) else [usage]
    price = model_price(model)
    for u in usages:
        line = f"  {label}: {u.input_tokens:,} in + {u.output_tokens:,} out"
        if price:
            line += f" = ${u.cost(*price):.4f}"
        else:
            line += f" (no price on record for {model})"
        print(line)


def load_personas():
    """Personas from disk, or None if the graph stage has not run yet."""
    from ragas.testset.persona import Persona

    if not PERSONAS_PATH.is_file():
        return None
    return [Persona(**p) for p in json.loads(PERSONAS_PATH.read_text())]


def build_personas(kg, llm, num_personas: int):
    """Generate and persist the personas the synthesizers ask questions as.

    A persona is just {name, role_description}. ragas clusters node summaries
    by embedding, invents one persona per cluster, matches personas to themes,
    and then conditions every question on (persona, term, style, length) -
    "create a question that aligns with the persona's perspective". It is a
    diversity mechanism: without it the whole set reads in one voice.

    Persisted because `generate()` would otherwise invent new ones on every
    run, and different personas mean different questions - so a regenerated
    test set would not be the same test set.
    """
    from ragas.testset.persona import generate_personas_from_kg

    personas = generate_personas_from_kg(
        kg=kg, llm=llm, num_personas=num_personas, callbacks=[]
    )
    PERSONAS_PATH.write_text(
        json.dumps([p.model_dump() for p in personas], indent=2) + "\n"
    )
    return personas


def local_embeddings():
    """bge-small on the host, the same model search uses. No API cost."""
    from config import load
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from process.embedder import MODELS
    from ragas.embeddings import LangchainEmbeddingsWrapper

    cfg = load()
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=MODELS[cfg.embedder.model]["hf_name"])
    )


def main():
    parser = argparse.ArgumentParser(description="Build the eval question set")
    parser.add_argument(
        "command", choices=["seed", "kg", "clusters", "personas", "generate"]
    )
    parser.add_argument("--per-topic", type=int, default=SEED_PER_TOPIC)
    parser.add_argument("--size", type=int, default=TESTSET_SIZE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    TESTSET_DIR.mkdir(parents=True, exist_ok=True)

    # Before any LangChain import runs, so ragas' calls are instrumented.
    # No-op unless PHOENIX_COLLECTOR_ENDPOINT is set, which is what keeps
    # this runnable with no collector listening.
    from observability import setup_observability

    setup_observability("eval-testset")

    if args.command == "seed":
        pool = seed_pool(args.per_topic)
        write_seed_pool(pool)
        print(f"{len(pool)} chunks -> {SEED_PATH}")
        return

    if args.command == "kg":
        from config import load

        pool = read_seed_pool()
        handler = cost_handler()
        llm, model = generator_llm(handler)
        r = load().ragas
        print(
            f"Building knowledge graph over {len(pool)} nodes with {model} "
            f"({r.max_workers} workers, {r.max_retries} retries)"
        )
        kg = build_kg(pool, llm, local_embeddings())
        kg.save(str(KG_PATH))
        print(f"{len(kg.nodes)} nodes, {len(kg.relationships)} relationships -> {KG_PATH}")
        for name, n in cluster_counts(kg).items():
            print(f"  {name:<22} {n} clusters")

        # Personas here rather than inside generate(): they are derived from
        # the graph, and persisting them is what makes a regenerated test set
        # the same test set.
        personas = build_personas(kg, llm, r.num_personas)
        print(f"{len(personas)} personas -> {PERSONAS_PATH}")
        for pers in personas:
            print(f"  {pers.name}: {pers.role_description}")

        print("\nSpend:")
        report_spend(handler, model, "graph + personas")
        return

    from ragas.testset.graph import KnowledgeGraph

    kg = KnowledgeGraph.load(str(KG_PATH))

    if args.command == "clusters":
        for name, n in assert_clusters(kg).items():
            print(f"  {name:<22} {n} clusters")
        return

    if args.command == "personas":
        # Separate from `kg` so a graph that survived a crashed run does not
        # have to be rebuilt just to get its personas.
        from config import load

        handler = cost_handler()
        llm, model = generator_llm(handler)
        personas = build_personas(kg, llm, load().ragas.num_personas)
        print(f"{len(personas)} personas -> {PERSONAS_PATH}")
        for pers in personas:
            print(f"  {pers.name}: {pers.role_description}")
        print("\nSpend:")
        report_spend(handler, model, "personas")
        return

    assert_clusters(kg)
    pool = read_seed_pool()
    handler = cost_handler()
    llm, model = generator_llm(handler)

    from config import load
    from ragas.testset.synthesizers.generate import TestsetGenerator

    r = load().ragas
    personas = load_personas()
    if personas:
        print(f"Using {len(personas)} saved personas")
    else:
        print("No saved personas - ragas will invent new ones for this run")

    generator = TestsetGenerator(
        llm=llm,
        embedding_model=local_embeddings(),
        knowledge_graph=kg,
        persona_list=personas,
    )
    testset = generator.generate(
        testset_size=args.size,
        query_distribution=query_distribution(llm),
        num_personas=r.num_personas,
        run_config=run_config(),
    )

    rows, unmapped = to_rows(testset, pool)
    GENERATED_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} questions -> {GENERATED_PATH}")
    if unmapped:
        print(f"WARNING: {unmapped} reference contexts did not map to a chunk id")

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["synthesizer"]] = counts.get(r["synthesizer"], 0) + 1
    for name, n in sorted(counts.items()):
        print(f"  {name:<40} {n}")

    by_topic: dict[str, int] = {}
    for row in rows:
        for topic in row["topics"] or ["(unmapped)"]:
            by_topic[topic] = by_topic.get(topic, 0) + 1
    print("\nQuestions per topic:")
    for topic, n in sorted(by_topic.items()):
        print(f"  {topic:<14} {n}")

    print("\nSpend:")
    report_spend(handler, model, "synthesis")


if __name__ == "__main__":
    main()
