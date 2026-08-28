"""Pipeline status query, shared between the REST /status endpoint and the
orchestrator's get_status tool - one source of truth for what "pending" means
at each stage, instead of the SQL living inside a langchain tool wrapper.
"""
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from config import load
from db.connection import PostgresInterface
from db.models import Chunk, ChunkEmbedding, Document, Object
from process.embedder import MODELS


def pipeline_status() -> dict:
    """Counts of documents/objects/chunks at each pipeline stage."""
    cfg = load()
    model_hf_name = MODELS[cfg.embedder.model]["hf_name"]

    with Session(PostgresInterface.connect()) as session:
        documents_total = session.execute(
            select(func.count()).select_from(Document)
        ).scalar_one()

        pending_download = session.execute(
            select(func.count())
            .select_from(Document)
            .where(Document.pdf_url.is_not(None))
            .where(Document.pdf_url != "")
            .where(~Document.id.in_(select(Object.doc_id)))
        ).scalar_one()

        objects_by_status = dict(
            session.execute(
                select(Object.status, func.count()).group_by(Object.status)
            ).all()
        )

        # Approximation: a chunk with *no* embedding row at all (any model)
        # counts as pending. Good enough for a status snapshot; doesn't
        # distinguish "never embedded" from "embedded with a different model".
        chunks_pending_embed = session.execute(
            select(func.count())
            .select_from(Chunk)
            .where(~Chunk.id.in_(select(ChunkEmbedding.chunk_id)))
        ).scalar_one()

    return {
        "documents_total": documents_total,
        "pending_download": pending_download,
        "objects_by_status": objects_by_status,
        "chunks_pending_embed": chunks_pending_embed,
        "embed_model": model_hf_name,
    }
