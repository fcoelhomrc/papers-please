"""Turns thumbs-up feedback into proposed eval/dataset.jsonl rows.

    uv run python -m eval.feedback              # print proposals
    uv run python -m eval.feedback --out new.jsonl

The eval set is hand-authored in eval/fixtures.py, which is the slowest way
imaginable to grow labels while every search someone runs is a labelling
opportunity going to waste. A thumbs-up on a result is exactly the judgement
`relevant_source_ids` encodes: "for this question, this paper is relevant".

Deliberately *proposes* rather than appends. Two reasons, and neither is
squeamishness:

  - A dataset row also needs `ground_truth`, which a thumb cannot supply.
    Every proposal comes out with that field blank for a human to fill, and
    a row with an invented ground truth would poison every faithfulness
    score computed against it thereafter.
  - Feedback is unvetted input. Silently appending would let a stray click
    move the numbers this project steers by, with no diff to notice it in.

Reads the database directly rather than through the API: this is a
maintenance script run beside eval.run, and the API would only be a slower
route to the same tables.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"


def _existing_questions(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    questions = set()
    for line in path.read_text().splitlines():
        if line.strip():
            questions.add(json.loads(line)["question"].strip().lower())
    return questions


def collect(session, min_votes: int = 1) -> list[dict]:
    """Group thumbs-up rows into one proposal per question.

    Keyed on the normalised question so the same search run twice with
    different results contributes both papers to one row, rather than two
    rows that disagree about what is relevant to the same question.

    A thumbs-*down* is not the inverse label. "This result is bad" says
    nothing about which paper would have been good, and `relevant_source_ids`
    has no way to record a negative - so downs are counted for reporting and
    otherwise left alone.
    """
    from db.models import Document, Feedback

    rows = session.execute(
        select(Feedback).where(Feedback.verdict == "up").order_by(Feedback.created_at)
    ).scalars().all()

    doc_ids = {r.doc_id for r in rows if r.doc_id is not None}
    source_by_doc = dict(
        session.execute(
            select(Document.id, Document.source_id).where(Document.id.in_(doc_ids))
        ).all()
    ) if doc_ids else {}

    by_question: dict[str, dict] = {}
    votes: dict[str, int] = defaultdict(int)
    for r in rows:
        key = r.query.strip().lower()
        votes[key] += 1
        entry = by_question.setdefault(
            key, {"question": r.query.strip(), "source_ids": [], "kinds": set()}
        )
        entry["kinds"].add(r.kind)
        # A paper marked relevant twice for one question is still one label.
        source_id = source_by_doc.get(r.doc_id)
        if source_id and source_id not in entry["source_ids"]:
            entry["source_ids"].append(source_id)

    return [
        {
            "question": e["question"],
            "relevant_source_ids": e["source_ids"],
            "votes": votes[key],
            "kinds": sorted(e["kinds"]),
        }
        for key, e in by_question.items()
        if votes[key] >= min_votes and e["source_ids"]
    ]


def to_dataset_rows(proposals: list[dict], skip: set[str]) -> list[dict]:
    """Dataset rows, with ground_truth left blank for a human.

    `category`/`domain` are what stratified sampling groups on
    (eval/run.py:stratified_sample), so a row missing them would quietly
    land in an unnamed stratum. "grounded"/"feedback" is honest: these
    questions came from use, not from a curated domain sweep.
    """
    return [
        {
            "category": "grounded",
            "domain": "feedback",
            "question": p["question"],
            "ground_truth": "",
            "relevant_source_ids": p["relevant_source_ids"],
        }
        for p in proposals
        if p["question"].strip().lower() not in skip
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-votes",
        type=int,
        default=1,
        help="only propose questions with at least this many thumbs-up",
    )
    parser.add_argument(
        "--out",
        help="write the proposals to this file instead of stdout. Never "
        "eval/dataset.jsonl itself - review, fill in ground_truth, then merge.",
    )
    args = parser.parse_args()

    from db.connection import PostgresInterface

    with Session(PostgresInterface.connect()) as session:
        proposals = collect(session, min_votes=args.min_votes)

    skip = _existing_questions(DATASET_PATH)
    rows = to_dataset_rows(proposals, skip)

    if not rows:
        print(
            f"{len(proposals)} question(s) with thumbs-up feedback, "
            f"{len(proposals) - len(rows)} already in the dataset — nothing new to propose."
        )
        return

    text = "\n".join(json.dumps(r) for r in rows)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"{len(rows)} proposed row(s) -> {args.out}")
    else:
        print(text)
    print(
        f"\n# {len(rows)} proposal(s). ground_truth is blank on purpose - a thumb "
        f"says which paper is relevant, not what the answer is. Fill it in, then "
        f"append to eval/dataset.jsonl."
    )


if __name__ == "__main__":
    main()
