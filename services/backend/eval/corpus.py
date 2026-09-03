"""The evaluation corpus: five adjacent topics, staged for human curation.

    uv run python -m eval.corpus stage          # fetch 40 candidates/topic
    uv run python -m eval.corpus status         # counts per topic and state

Why five topics and not twenty
------------------------------
Retrieval difficulty comes from *within-topic* neighbours. A hundred papers
spread over twenty unrelated fields is trivial - lexical overlap alone
separates "protein folding" from "convex optimization", every configuration
scores near 1.0, and the ablations discriminate nothing. Hard negatives are
papers that share vocabulary and differ only in the detail a question asks
about.

The five below deliberately overlap: they share half their terminology, cite
each other, and in several cases describe competing solutions to the same
problem. That is the point.

It also has a mechanical consequence. The multi-hop synthesizers in
`eval.testset` need graph edges built from summary similarity (cosine >= 0.5)
and shared entities. On a topically scattered corpus no pair clears that
threshold, ragas silently drops both multi-hop synthesizers, and the result is
an all-single-hop test set with no error to indicate it.

Why 40 candidates for 20 slots
------------------------------
Taking the top 20 by citation produces a *list*, not a corpus. The reviewer
keeps near-duplicates on purpose - competing methods, v1/v2 of an idea - and
mixes short workshop papers with long surveys so chunk-count skew exercises
the neighbour window and the reranker. That needs more choice than slots.

Candidates land as corpus='candidate', which `PdfFetcher.pending()` refuses to
download, so nothing is OCR'd until a human has promoted it.
"""
import argparse
import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# (slug, query). Queries are broad on purpose - narrowing them costs the
# within-topic variety that makes hard negatives.
TOPICS: list[tuple[str, str]] = [
    ("retrieval", "retrieval augmented generation"),
    ("agents", "large language model agents tool use"),
    ("evaluation", "evaluating large language models benchmark"),
    # Broader than "efficient llm inference quantization", which matched only
    # 28 arXiv papers - the bulk endpoint ANDs the terms, so a four-word query
    # narrows hard. This one returns 123 and still stays on topic.
    ("efficiency", "large language model quantization"),
    ("alignment", "language model alignment reinforcement learning human feedback"),
]

CANDIDATES_PER_TOPIC = 40
KEEP_PER_TOPIC = 20

# 2023 onwards: the topics barely existed before, and older papers drag in
# survey-of-a-different-field noise. minCitationCount filters the long tail of
# preprints that were never read; sorting by citations puts the papers a
# reviewer will recognise at the top of the list.
YEAR_FROM = "2023-"
MIN_CITATIONS = 10
SORT = "citationCount:desc"


def stage_candidates(per_topic: int = CANDIDATES_PER_TOPIC) -> dict[str, int]:
    """Fetch candidates for every topic. Returns new papers per topic.

    Idempotent: source_id is UNIQUE and the insert does nothing on conflict,
    so re-running adds only papers not already known.
    """
    from ingest.fetcher import SemanticScholarFetcher

    fetcher = SemanticScholarFetcher()
    added = {}
    for slug, query in TOPICS:
        added[slug] = fetcher.fetch(
            query=query,
            max_papers=per_topic,
            year=YEAR_FROM,
            open_access_pdf=True,
            min_citations=MIN_CITATIONS,
            sort=SORT,
            arxiv_only=True,
            corpus="candidate",
            topic=slug,
        )
        logger.info(f"{slug}: {added[slug]} new candidates")
    return added


def counts() -> list[tuple[str, str, int]]:
    """(topic, corpus, n) across the staged and promoted corpus."""
    from db.connection import PostgresInterface
    from db.models import Document

    stmt = (
        select(Document.topic, Document.corpus, func.count(Document.id))
        .where(Document.topic.is_not(None))
        .group_by(Document.topic, Document.corpus)
        .order_by(Document.topic, Document.corpus)
    )
    with Session(PostgresInterface.connect()) as session:
        return [(t, c, n) for t, c, n in session.execute(stmt).all()]


def main():
    parser = argparse.ArgumentParser(description="Stage the eval corpus")
    parser.add_argument("command", choices=["stage", "status"])
    parser.add_argument("--per-topic", type=int, default=CANDIDATES_PER_TOPIC)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "stage":
        added = stage_candidates(args.per_topic)
        print(f"\nStaged {sum(added.values())} new candidates across {len(added)} topics")

    for topic, corpus, n in counts():
        print(f"  {topic:<12} {corpus:<10} {n}")


if __name__ == "__main__":
    main()
