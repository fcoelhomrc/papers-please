"""Throw away everything derived from the PDFs and let the workers rebuild it.

    uv run python -m scripts.reset_derived            # dry run, prints the damage
    uv run python -m scripts.reset_derived --yes      # actually do it

Papers are kept. `documents` carries the corpus curation - 200 candidates
reviewed down to 100, the expensive human work - and `objects` points at PDFs
already on the papers_data volume. Neither is touched, so nothing is
re-fetched and nothing is re-reviewed.

Chunks, embeddings and Pinecone vectors are thrown away, because the chunker
now records `element_type` and there is no way to backfill it: docling's
TripletTableSerializer flattens a table into prose before the chunker sees it,
so the label has to be captured during conversion or not at all.

**There is no backfill logic here on purpose.** Emptying the tables is enough,
because both workers already select their own work:

  - `PdfChunker.pending()` takes `objects.status == 'pending'`
  - `PdfEmbedder.pending()` takes chunks not in `chunk_embeddings`

so setting the status back and truncating the tables puts the pipeline in
exactly the state it was in after download, and it refills itself.

Chunk ids are renumbered by this. Any test set holding `reference_chunk_ids`
is invalidated and must be regenerated - which is why this runs before the
test-set rebuild, not after.
"""
import argparse
import logging
import os

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from db.connection import PostgresInterface
from db.models import Chunk, ChunkEmbedding, Object

logger = logging.getLogger(__name__)

# Only 'chunked' is safe to reset. 'downloading' means the fetcher has a row
# but the file may not have landed yet (ingest/fetcher.py:239), so flipping it
# to 'pending' would hand the chunker a path that does not exist. 'failed' and
# 'dead' are left for the existing requeue loop to reason about.
RESETTABLE = "chunked"


def survey(session) -> dict:
    by_status = dict(
        session.execute(select(Object.status, func.count()).group_by(Object.status)).all()
    )
    return {
        "chunks": session.scalar(select(func.count()).select_from(Chunk)),
        "embeddings": session.scalar(select(func.count()).select_from(ChunkEmbedding)),
        "objects_by_status": by_status,
    }


def add_columns(session) -> None:
    """The two new columns, for a volume that already exists.

    schema.sql carries them for a fresh volume; this is the same change for
    the running database, and is a no-op on a database that already has them.
    """
    session.execute(
        text(
            "ALTER TABLE chunks "
            "ADD COLUMN IF NOT EXISTS heading_path TEXT, "
            "ADD COLUMN IF NOT EXISTS element_type TEXT"
        )
    )


def wipe_postgres(session) -> int:
    # CASCADE reaches chunk_embeddings, which references chunks. embedding_models
    # is deliberately left alone so model_id stays stable across the rebuild.
    session.execute(text("TRUNCATE chunks CASCADE"))
    return session.execute(
        text("UPDATE objects SET status = 'pending', attempts = 0 WHERE status = :s"),
        {"s": RESETTABLE},
    ).rowcount


def wipe_pinecone(namespace: str, index_name: str) -> str:
    from pinecone.grpc import PineconeGRPC as Pinecone

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if not pc.has_index(index_name):
        return f"index {index_name!r} does not exist - nothing to wipe"
    index = pc.Index(index_name)
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        # Pinecone raises rather than no-opping when the namespace has never
        # held a vector, which is a success for our purposes.
        return f"namespace {namespace!r} not deleted ({type(e).__name__}) - treating as empty"
    return f"namespace {namespace!r} emptied"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes", action="store_true", help="actually destroy things (default: dry run)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from config import load
    from process.embedder import MODELS

    cfg = load()
    namespace = cfg.search.namespace
    index_name = MODELS[cfg.embedder.model]["index_name"]

    db = PostgresInterface()
    with Session(db.engine) as session:
        before = survey(session)
        print("\ncurrent state")
        print(f"  chunks              {before['chunks']:,}")
        print(f"  chunk_embeddings    {before['embeddings']:,}")
        for status, n in sorted(before["objects_by_status"].items()):
            mark = " -> pending" if status == RESETTABLE else " (left alone)"
            print(f"  objects {status:<12} {n:>5}{mark}")
        print(f"\n  pinecone index      {index_name} / namespace {namespace!r}")

        if not args.yes:
            print("\ndry run - nothing changed. Re-run with --yes to destroy.")
            return

        add_columns(session)
        reset = wipe_postgres(session)
        session.commit()
        print(f"\ntruncated chunks and chunk_embeddings; {reset} objects set to pending")

    print(wipe_pinecone(namespace, index_name))

    with Session(db.engine) as session:
        after = survey(session)
    assert after["chunks"] == 0, f"chunks not empty: {after['chunks']}"
    assert after["embeddings"] == 0, f"embeddings not empty: {after['embeddings']}"
    print("\ndone. Start the chunk and embed workers to refill.")


if __name__ == "__main__":
    main()
