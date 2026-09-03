"""Backing store for the human review passes.

Three things get reviewed on the way to a trustworthy number, and all three
are human work that ragas does not remove:

  1. **Corpus curation** - 40 staged candidates per topic, 20 kept. Taking the
     top 20 by citation would produce a list, not a corpus; the reviewer keeps
     near-duplicates on purpose and mixes paper lengths.
  2. **Question review** - the generated test set, ~20-30% of which is
     unusable.
  3. **Judge validation** - human verdicts on statements, to get a kappa.

Only (1) is implemented here so far; it is what unblocks the ingest.

Corpus state lives in Postgres rather than a JSON file, because `corpus` is
already the column every retrieval path filters on - a second source of truth
would let the UI and the retriever disagree about what the corpus is.
"""
import logging

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from db.connection import PostgresInterface
from db.models import Document

logger = logging.getLogger(__name__)

CANDIDATE = "candidate"
EVAL = "eval"
REJECTED = "main"

# What a PATCH may set a candidate to. 'main' is the reject state rather than
# a delete: a rejected paper stays fetched, so re-running the stage command
# does not re-offer it, and the rejection survives.
#
# 'reset' exists because curation is a long manual pass and a misclick must
# not be one-way. Without it a rejected paper drops out of the list and the
# only route back is a hand-written UPDATE.
DECISIONS = {"keep": EVAL, "reject": REJECTED, "reset": CANDIDATE}


def candidates(topic: str | None = None) -> list[dict]:
    """Staged and already-kept papers, newest-cited first.

    Every topic-tagged paper comes back, in all three states. Kept ones
    because the judgement is comparative - whether to keep this paper
    depends on what is already in the topic - and rejected ones because a
    decision that cannot be seen cannot be undone.
    """
    stmt = (
        select(
            Document.id,
            Document.source_id,
            Document.title,
            Document.abstract,
            Document.authors,
            Document.year,
            Document.citation_count,
            Document.corpus,
            Document.topic,
            Document.pdf_url,
        )
        .where(Document.topic.is_not(None))
        .order_by(Document.citation_count.desc().nullslast(), Document.id)
    )
    if topic:
        stmt = stmt.where(Document.topic == topic)

    with Session(PostgresInterface.connect()) as session:
        rows = session.execute(stmt).all()
    return [dict(r._mapping) for r in rows]


def topic_summary() -> list[dict]:
    """Per-topic counts of what is staged, kept and rejected.

    Rejected is counted by topic rather than by corpus alone: a rejected
    candidate becomes corpus='main', which is also where every ad-hoc fetch
    lands, so only the topic tag distinguishes the two.
    """
    stmt = (
        select(Document.topic, Document.corpus, func.count(Document.id))
        .where(Document.topic.is_not(None))
        .group_by(Document.topic, Document.corpus)
    )
    with Session(PostgresInterface.connect()) as session:
        rows = session.execute(stmt).all()

    by_topic: dict[str, dict] = {}
    for topic, corpus, n in rows:
        entry = by_topic.setdefault(
            topic, {"topic": topic, "candidates": 0, "kept": 0, "rejected": 0}
        )
        if corpus == CANDIDATE:
            entry["candidates"] = n
        elif corpus == EVAL:
            entry["kept"] = n
        else:
            entry["rejected"] = n
    return sorted(by_topic.values(), key=lambda e: e["topic"])


def decide(doc_id: int, decision: str) -> str:
    """Promote or reject one candidate. Returns the new corpus value.

    Raises ValueError for an unknown decision or a doc that is not part of a
    topic, so a typo in the API surfaces as a 400 rather than silently
    moving an unrelated paper into the eval corpus.
    """
    if decision not in DECISIONS:
        raise ValueError(f"unknown decision {decision!r}")
    target = DECISIONS[decision]

    with Session(PostgresInterface.connect()) as session:
        result = session.execute(
            update(Document)
            .where(Document.id == doc_id)
            .where(Document.topic.is_not(None))
            .values(corpus=target)
            .returning(Document.id)
        )
        if result.first() is None:
            session.rollback()
            raise ValueError(f"no staged candidate with id {doc_id}")
        session.commit()
    return target
