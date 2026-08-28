"""Seeds eval/fixtures.py's documents into Postgres + Pinecone if they're
not already there - so `eval/run.py` produces reproducible scores
regardless of the dev DB's current state (manually fetched papers, a wipe,
a fresh clone). Idempotent: only inserts fixtures whose source_id isn't
already present.
"""
import logging

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from db.connection import PostgresInterface
from db.models import Chunk, Document, Object
from eval.fixtures import FIXTURES
from process.embedder import PdfEmbedder

logger = logging.getLogger(__name__)


def ensure_fixtures_seeded() -> None:
    engine = PostgresInterface.connect()

    with Session(engine) as session:
        existing = set(
            session.execute(
                select(Document.source_id).where(
                    Document.source_id.in_([f["source_id"] for f in FIXTURES])
                )
            )
            .scalars()
            .all()
        )
        missing = [f for f in FIXTURES if f["source_id"] not in existing]
        if not missing:
            logger.info("eval fixtures already present, nothing to seed")
            return

        for fixture in missing:
            doc_id = session.execute(
                insert(Document)
                .values(source_id=fixture["source_id"], title=fixture["title"])
                .on_conflict_do_nothing(index_elements=["source_id"])
                .returning(Document.id)
            ).scalar_one()
            obj_id = session.execute(
                insert(Object)
                .values(
                    doc_id=doc_id,
                    path=f"eval-fixtures/{fixture['source_id']}.pdf",
                    status="chunked",
                )
                .returning(Object.id)
            ).scalar_one()
            for i, text in enumerate(fixture["chunks"]):
                session.execute(
                    insert(Chunk).values(
                        obj_id=obj_id, chunk_index=i, chunk_text=text, page_num=1
                    )
                )
        session.commit()

    logger.info(f"seeded {len(missing)} eval fixture documents, embedding now")

    # Reuse the real embedding pipeline rather than reimplementing
    # embed+upsert - PdfEmbedder.pending() picks up exactly the chunks we
    # just inserted (chunk_text set, no chunk_embeddings row yet) and
    # nothing else, as long as the rest of the pipeline has already caught
    # up on unrelated chunks.
    PdfEmbedder().execute()
