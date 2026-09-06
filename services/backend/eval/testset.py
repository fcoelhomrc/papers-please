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
import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
TESTSET_DIR = HERE / "testset"
GENERATED_PATH = TESTSET_DIR / "generated.jsonl"

# Generation runs once per topic, against a graph built only from that topic's
# chunks. v1 did it in one pass over a pooled graph and the result was 7:1
# skewed - 75 of 131 questions on `agents` against 13 on `alignment` - from a
# seed pool that was balanced 80/80/80/80/80. Curation was even-handed across
# topics (62-87% keep rates), so the skew was ragas': the synthesizers draw
# from whichever clusters are richest, and `agents` had the most. A per-topic
# quota makes balance structural instead of hoped for.
SEED_PER_TOPIC = 100
PER_TOPIC_SIZE = 30
TESTSET_SIZE = 150


def topic_slugs() -> list[str]:
    from eval.corpus import TOPICS

    return [slug for slug, _ in TOPICS]


def seed_path(topic: str) -> Path:
    return TESTSET_DIR / f"seed_pool-{topic}.jsonl"


def kg_path(topic: str) -> Path:
    return TESTSET_DIR / f"kg-{topic}.json"


def personas_path(topic: str) -> Path:
    return TESTSET_DIR / f"personas-{topic}.json"


def generated_path(topic: str) -> Path:
    return TESTSET_DIR / f"generated-{topic}.jsonl"

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

# Multi-hop synthesizers prefix each retrieved context with the hop it came
# from - "<1-hop>\n\n", "<2-hop>\n\n" - so the string is the marker plus
# the node text, and an exact lookup misses every multi-hop context. On a
# 7-question trial this silently emptied `reference_chunk_ids` for all four
# multi-hop rows: 8 of 11 contexts unresolved, which downstream reads as
# "retrieval found nothing" on every run forever.
HOP_MARKER = re.compile(r"^<\d+-hop>\s*\n+")

# 50/25/25 rather than ragas' default third each. Single-hop questions are the
# ones a retrieval ablation can actually discriminate on; multi-hop are the
# ones that catch a pipeline that retrieves one good chunk and stops.
DISTRIBUTION = {
    "single_hop_specific": 0.50,
    "multi_hop_specific": 0.25,
    "multi_hop_abstract": 0.25,
}


# --- Stage 1: seed pool ----------------------------------------------------


def seed_pool(
    per_topic: int = SEED_PER_TOPIC, seed: int = SEED, topic: str | None = None
) -> list[dict]:
    """A stratified sample of eval-corpus chunks, `per_topic` from each topic.

    Stratified rather than random over the whole corpus because chunk counts
    differ several-fold between papers - a 30-page survey would otherwise
    crowd out an entire topic of short papers, and the question set would
    inherit that skew.

    `topic` narrows it to one, which is how generation is run: a graph per
    topic, a quota per topic. Note this samples the *seed* pool only - the
    haystack retrieval searches is still every chunk in the corpus, so a
    question seeded from one topic is still answered against all five.
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
    if topic is not None:
        stmt = stmt.where(Document.topic == topic)
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


def write_seed_pool(pool: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(c) + "\n" for c in pool))
    return path


def read_seed_pool(path: Path) -> list[dict]:
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


def choose_threshold(counts: dict[float, int], cap: int = 1500) -> float:
    """The most generous threshold that still leaves a tractable graph.

    Lower threshold means more edges means more clusters to draw multi-hop
    questions from - up to the point where `find_indirect_clusters` stops
    returning. So: take the smallest candidate whose edge count is still under
    the cap, and fall back to the strictest if even that overflows.

    1500 is above the 788 that clustered in 12.9s on the pooled graph and well
    under the 4,047 that timed out.
    """
    affordable = [t for t in sorted(counts) if counts[t] <= cap]
    return affordable[0] if affordable else max(counts)


def relationship_transforms(threshold: float = COSINE_THRESHOLD):
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
                threshold=threshold,
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


def build_kg(pool: list[dict], llm, embeddings, threshold: float | None = None):
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

    # Measured on this graph rather than assumed: within one topic every
    # summary resembles every other more closely than it did in the pooled
    # v1 graph, so the threshold that worked there can produce a complete
    # graph here - and clustering a complete graph does not return.
    counts = measure_thresholds(kg)
    if threshold is None:
        threshold = choose_threshold(counts)
    logger.info(
        "summary_similarity edges by threshold: "
        + ", ".join(f"{t}->{n}" for t, n in sorted(counts.items(), reverse=True))
        + f" | using {threshold}"
    )
    apply_transforms(kg, relationship_transforms(threshold), run_config())
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


def measure_thresholds(kg, candidates=(0.90, 0.85, 0.80, 0.75, 0.70)) -> dict[float, int]:
    """How many summary_similarity edges each cosine threshold would create.

    COSINE_THRESHOLD = 0.80 was measured on the *pooled* 400-node graph. Within
    a single topic every summary resembles every other more closely, so the
    same threshold can produce a near-complete graph - and a near-complete
    graph is what made `find_indirect_clusters` hang for 13 minutes at 0% CPU
    before it was killed. Measured on the pooled graph, the relationship count
    is the leading indicator:

        126 rels ->   394 clusters in  0.1s
        788 rels -> 8,341 clusters in 12.9s
      4,047 rels -> timed out at 60s

    So this counts edges rather than clustering, which is fast and cannot
    hang. Aim for a few hundred: far more clusters than 30 questions need,
    and still tractable.
    """
    import numpy as np

    vectors = [
        n.properties.get("summary_embedding")
        for n in kg.nodes
        if n.properties.get("summary_embedding") is not None
    ]
    if len(vectors) < 2:
        return {t: 0 for t in candidates}

    matrix = np.asarray(vectors, dtype=float)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    similarity = matrix @ matrix.T
    # Upper triangle only: the builder makes one edge per unordered pair, and
    # counting both halves plus the diagonal would overstate it by ~2x.
    pairs = similarity[np.triu_indices(len(matrix), k=1)]
    return {t: int((pairs >= t).sum()) for t in candidates}


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

    The one transformation ragas does apply is the multi-hop marker, which
    is stripped first - see HOP_MARKER.

    A context that still does not resolve is dropped and counted by the
    caller, because a silently empty `reference_chunk_ids` reads as
    "retrieval found nothing" on every run forever.
    """
    return [
        lookup[key]
        for key in (HOP_MARKER.sub("", c) for c in reference_contexts)
        if key in lookup
    ]


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
    chat = openrouter_chat(model, cfg.ragas.generator_max_tokens, cfg)
    if cfg.ragas.reasoning_effort:
        # Provider-specific, so it rides in extra_body rather than a named
        # ChatOpenAI argument.
        chat.extra_body = {"reasoning": {"effort": cfg.ragas.reasoning_effort}}
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


def load_personas(path: Path):
    """Personas from disk, or None if the graph stage has not run yet."""
    from ragas.testset.persona import Persona

    if not path.is_file():
        return None
    return [Persona(**p) for p in json.loads(path.read_text())]


def build_personas(kg, llm, num_personas: int, path: Path):
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
    path.write_text(json.dumps([p.model_dump() for p in personas], indent=2) + "\n")
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




# --- Per-topic driver ------------------------------------------------------


def merge_generated(topics: list[str]) -> list[dict]:
    """Concatenate the per-topic files into one, renumbering ids.

    Ids are assigned here rather than per topic so `q0007` means one question
    across the whole set - the review UI and every result file key on them, and
    two topics both owning `q0007` would silently merge two questions' review
    state.
    """
    rows: list[dict] = []
    for topic in topics:
        path = generated_path(topic)
        if not path.is_file():
            logger.warning(f"no generated file for {topic} - skipping")
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    for i, row in enumerate(rows):
        row["id"] = f"q{i:04d}"
    GENERATED_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def topic_report(rows: list[dict]) -> None:
    by_topic: dict[str, int] = {}
    by_synth: dict[str, int] = {}
    for row in rows:
        for topic in row["topics"] or ["(unmapped)"]:
            by_topic[topic] = by_topic.get(topic, 0) + 1
        by_synth[row["synthesizer"]] = by_synth.get(row["synthesizer"], 0) + 1

    print("\nQuestions per topic:")
    for topic, n in sorted(by_topic.items()):
        print(f"  {topic:<14} {n}")
    print("\nQuestions per synthesizer:")
    for name, n in sorted(by_synth.items()):
        print(f"  {name:<40} {n}")


def stage_seed(topic: str, per_topic: int) -> None:
    pool = seed_pool(per_topic, topic=topic)
    write_seed_pool(pool, seed_path(topic))
    print(f"[{topic}] {len(pool)} chunks -> {seed_path(topic)}")


def stage_kg(topic: str, threshold: float | None) -> None:
    from config import load

    pool = read_seed_pool(seed_path(topic))
    handler = cost_handler()
    llm, model = generator_llm(handler)
    r = load().ragas
    print(
        f"[{topic}] building graph over {len(pool)} nodes with {model} "
        f"({r.max_workers} workers, effort={r.reasoning_effort or 'default'})"
    )
    kg = build_kg(pool, llm, local_embeddings(), threshold=threshold)
    kg.save(str(kg_path(topic)))
    print(f"[{topic}] {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")
    for name, n in cluster_counts(kg).items():
        print(f"  {name:<22} {n} clusters")

    personas = build_personas(kg, llm, r.num_personas, personas_path(topic))
    print(f"[{topic}] {len(personas)} personas")
    for pers in personas:
        print(f"  {pers.name}: {pers.role_description}")
    report_spend(handler, model, f"{topic} graph + personas")


def stage_generate(topic: str, size: int) -> list[dict]:
    from config import load
    from ragas.testset.graph import KnowledgeGraph
    from ragas.testset.synthesizers.generate import TestsetGenerator

    kg = KnowledgeGraph.load(str(kg_path(topic)))
    assert_clusters(kg)
    pool = read_seed_pool(seed_path(topic))
    handler = cost_handler()
    llm, model = generator_llm(handler)
    r = load().ragas

    personas = load_personas(personas_path(topic))
    print(f"[{topic}] generating {size} questions from {len(kg.nodes)} nodes")

    generator = TestsetGenerator(
        llm=llm,
        embedding_model=local_embeddings(),
        knowledge_graph=kg,
        persona_list=personas,
    )
    testset = generator.generate(
        testset_size=size,
        query_distribution=query_distribution(llm),
        num_personas=r.num_personas,
        run_config=run_config(),
    )

    rows, unmapped = to_rows(testset, pool)
    generated_path(topic).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"[{topic}] {len(rows)} questions -> {generated_path(topic)}")
    if unmapped:
        print(f"[{topic}] WARNING: {unmapped} reference contexts did not map to a chunk id")
    report_spend(handler, model, f"{topic} synthesis")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Build the eval question set, one topic at a time"
    )
    parser.add_argument(
        "command",
        choices=["seed", "kg", "clusters", "thresholds", "generate", "merge", "all"],
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="one topic slug; default is every topic in eval/corpus.py",
    )
    parser.add_argument("--per-topic", type=int, default=SEED_PER_TOPIC)
    parser.add_argument(
        "--size", type=int, default=PER_TOPIC_SIZE, help="questions per topic"
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=None,
        help="override the per-topic measured threshold",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    TESTSET_DIR.mkdir(parents=True, exist_ok=True)

    # Before any LangChain import runs, so ragas' calls are instrumented.
    # No-op unless PHOENIX_COLLECTOR_ENDPOINT is set, which is what keeps
    # this runnable with no collector listening.
    from observability import setup_observability

    setup_observability("eval-testset")

    topics = [args.topic] if args.topic else topic_slugs()

    if args.command == "merge":
        topic_report(merge_generated(topics))
        print(f"\n-> {GENERATED_PATH}")
        return

    if args.command in ("clusters", "thresholds"):
        from ragas.testset.graph import KnowledgeGraph

        for topic in topics:
            kg = KnowledgeGraph.load(str(kg_path(topic)))
            print(f"\n[{topic}] {len(kg.nodes)} nodes")
            if args.command == "clusters":
                for name, n in assert_clusters(kg).items():
                    print(f"  {name:<22} {n} clusters")
            else:
                counts = measure_thresholds(kg)
                for t, n in sorted(counts.items(), reverse=True):
                    print(f"  {t:.2f} -> {n:>6} edges")
                print(f"  would use {choose_threshold(counts)}")
        return

    for topic in topics:
        if args.command in ("seed", "all"):
            stage_seed(topic, args.per_topic)
        if args.command in ("kg", "all"):
            stage_kg(topic, args.cosine_threshold)
        if args.command in ("generate", "all"):
            stage_generate(topic, args.size)

    if args.command in ("generate", "all"):
        topic_report(merge_generated(topics))
        print(f"\n-> {GENERATED_PATH}")


if __name__ == "__main__":
    main()
