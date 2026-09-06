"""Backing store for the human review passes.

Three things get reviewed on the way to a trustworthy number, and all three
are human work that ragas does not remove:

  1. **Corpus curation** - 40 staged candidates per topic, 20 kept. Taking the
     top 20 by citation would produce a list, not a corpus; the reviewer keeps
     near-duplicates on purpose and mixes paper lengths.
  2. **Question review** - the generated test set, ~20-30% of which is
     unusable.
  3. **Judge validation** - human verdicts on statements, to get a kappa.

(1) and (2) are implemented here; (3) follows once there are answers to judge.

The two use different stores, deliberately. Corpus state lives in Postgres
because `corpus` is already the column every retrieval path filters on, and a
second source of truth would let the UI and the retriever disagree about what
the corpus is. Question review lives in a JSON sidecar because the generated
set is a file: keeping decisions out of it leaves the generator's output
immutable, so a diff shows exactly what a human changed.
"""
import json
import logging
from pathlib import Path

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


# --- Question review -------------------------------------------------------

GENERATED_PATH = Path(__file__).parent / "testset" / "generated.jsonl"
REVIEW_PATH = Path(__file__).parent / "testset" / "review.json"

QUESTION_DECISIONS = ("keep", "drop", "undecided")


def load_generated(path: Path | None = None) -> list[dict]:
    # Resolved at call time, not bound as a default: a default argument is
    # evaluated once at import, so tests pointing the module constant at a
    # fixture directory would silently keep reading the real test set.
    path = path or GENERATED_PATH
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_review(path: Path | None = None) -> dict[str, dict]:
    """Review decisions, keyed by question id.

    A sidecar rather than edits to generated.jsonl, so the generator's output
    stays immutable: a diff then shows exactly what a human changed, and
    re-inspecting or regenerating never silently discards review work.
    """
    path = path or REVIEW_PATH
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def save_review(state: dict[str, dict], path: Path | None = None) -> None:
    path = path or REVIEW_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def review_question(
    qid: str,
    decision: str | None = None,
    question: str | None = None,
    reference: str | None = None,
    note: str | None = None,
) -> dict:
    """Record one review decision. Returns the stored entry.

    Editing a question without editing its reference is the trap this
    interface exists to make visible: `context_recall` grades the pipeline's
    answer against `reference`, so a reworded question with a stale gold
    answer silently marks correct answers wrong.
    """
    if decision is not None and decision not in QUESTION_DECISIONS:
        raise ValueError(f"unknown decision {decision!r}")
    if qid not in {q["id"] for q in load_generated()}:
        raise ValueError(f"no generated question with id {qid}")

    state = load_review()
    entry = state.get(qid, {})
    for field, value in (
        ("decision", decision),
        ("question", question),
        ("reference", reference),
        ("note", note),
    ):
        if value is not None:
            entry[field] = value
    state[qid] = entry
    save_review(state)
    return entry


def questions() -> list[dict]:
    """Generated questions with their review state merged in.

    `question` and `reference` carry the edited text where one exists, with
    the original kept alongside so the UI can show what changed rather than
    quietly replacing it.
    """
    state = load_review()
    merged = []
    for q in load_generated():
        entry = state.get(q["id"], {})
        merged.append(
            {
                **q,
                "decision": entry.get("decision", "undecided"),
                "note": entry.get("note", ""),
                "question": entry.get("question", q["question"]),
                "reference": entry.get("reference", q["reference"]),
                "original_question": q["question"],
                "original_reference": q["reference"],
                "edited": bool(entry.get("question") or entry.get("reference")),
            }
        )
    return merged


def question_summary(merged: list[dict]) -> dict:
    """Counts the reviewer steers by: progress, and the two distributions
    that decide whether the kept set is balanced."""
    kept = [q for q in merged if q["decision"] == "keep"]
    by_topic: dict[str, int] = {}
    by_synth: dict[str, int] = {}
    for q in kept:
        for topic in q["topics"] or ["(none)"]:
            by_topic[topic] = by_topic.get(topic, 0) + 1
        by_synth[q["synthesizer"]] = by_synth.get(q["synthesizer"], 0) + 1
    return {
        "total": len(merged),
        "kept": len(kept),
        "dropped": sum(1 for q in merged if q["decision"] == "drop"),
        "undecided": sum(1 for q in merged if q["decision"] == "undecided"),
        "kept_by_topic": by_topic,
        "kept_by_synthesizer": by_synth,
    }


CURATED_PATH = Path(__file__).parent / "testset" / "curated.jsonl"


def curated() -> list[dict]:
    """The kept questions, with edits applied. What the harness evaluates.

    Resolved from generated.jsonl plus review.json rather than read from a
    file the reviewer edits, so a review still in progress cannot change what
    an experiment measured halfway through it.
    """
    return [
        {
            "id": q["id"],
            "question": q["question"],
            "reference": q["reference"],
            "reference_contexts": q["reference_contexts"],
            "reference_chunk_ids": q["reference_chunk_ids"],
            "reference_doc_ids": q["reference_doc_ids"],
            "topics": q["topics"],
            "synthesizer": q["synthesizer"],
            "edited": q["edited"],
        }
        for q in questions()
        if q["decision"] == "keep"
    ]


def write_curated(path: Path | None = None) -> Path:
    path = path or CURATED_PATH
    rows = curated()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def load_curated(path: Path | None = None) -> list[dict]:
    path = path or CURATED_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found - run `python -m eval.review export` after curating"
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Curated eval question set")
    parser.add_argument("command", choices=["export", "status"])
    args = parser.parse_args()

    merged = questions()
    summary = question_summary(merged)
    if args.command == "export":
        if summary["undecided"]:
            print(f"WARNING: {summary['undecided']} questions still undecided")
        path = write_curated()
        print(f"{summary['kept']} curated questions -> {path}")

    for key in ("total", "kept", "dropped", "undecided"):
        print(f"  {key:<12} {summary[key]}")
    print("  by topic     ", dict(sorted(summary["kept_by_topic"].items())))
    print("  by type      ", dict(sorted(summary["kept_by_synthesizer"].items())))


if __name__ == "__main__":
    main()
