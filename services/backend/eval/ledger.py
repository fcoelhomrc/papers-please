"""Append-only record of every eval run, committed to the repo.

`eval/results/*.json` is gitignored scratch that gets cleaned up, and the
judged runs only ever existed as markdown - so there was no single place
that answered "what have we measured so far", and deleting scratch output
silently deleted results. The ledger is that place: one line per run,
carrying the run's config and its headline metrics, small enough to commit
and diff.

Deliberately not a database. It's a few dozen lines of JSON that a human
reads in a diff, and adding a schema migration story to a personal project's
eval bookkeeping would cost more than it returns.
"""
import json
from pathlib import Path

LEDGER_PATH = Path(__file__).parent / "ledger.jsonl"

# Metrics a run may carry. Judged runs have the first four, retrieval runs
# the rest; the table leaves the others blank rather than pretending a run
# measured something it didn't.
JUDGED_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
RETRIEVAL_METRICS = ("recall", "precision", "hit_rate", "mrr", "ndcg", "abstention_precision")


def load(path: Path | None = None) -> list[dict]:
    path = path or LEDGER_PATH
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(record: dict, path: Path | None = None) -> bool:
    """Add a run. Idempotent on `id`, so re-ingesting existing artifacts
    doesn't duplicate rows. Returns True if it was actually added."""
    path = path or LEDGER_PATH
    if "id" not in record:
        raise ValueError("ledger records need an 'id'")
    if any(r["id"] == record["id"] for r in load(path)):
        return False
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return True
