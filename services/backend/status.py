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

# Order the Queue page lists work in: what needs attention, then what is
# moving, then what is finished. Sorting by id or timestamp instead would
# bury a dead object under hundreds of completed ones, which is the opposite
# of what a queue view is for.
STATUS_ORDER = [
    "dead",
    "download_failed",
    "failed",
    "awaiting_download",
    "downloading",
    "pending",
    "chunked",
    "embedded",
]


def queue_items(limit: int = 50) -> list[dict]:
    """Recent papers and where each one has got to.

    Two queries, not one per document: the per-object chunk and embedding
    counts come back as grouped aggregates and are joined in Python, which
    keeps this O(1) round trips no matter how long the list is.
    """
    with Session(PostgresInterface.connect()) as session:
        rows = session.execute(
            select(
                Document.id,
                Document.title,
                Document.pdf_url,
                Object.id.label("obj_id"),
                Object.status,
                Object.attempts,
            )
            .outerjoin(Object, Object.doc_id == Document.id)
            .order_by(Document.created_at.desc())
            .limit(limit)
        ).all()

        obj_ids = [r.obj_id for r in rows if r.obj_id is not None]
        chunk_counts: dict[int, int] = {}
        embedded_counts: dict[int, int] = {}
        if obj_ids:
            chunk_counts = dict(
                session.execute(
                    select(Chunk.obj_id, func.count())
                    .where(Chunk.obj_id.in_(obj_ids))
                    .group_by(Chunk.obj_id)
                ).all()
            )
            embedded_counts = dict(
                session.execute(
                    select(Chunk.obj_id, func.count())
                    .select_from(Chunk)
                    .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
                    .where(Chunk.obj_id.in_(obj_ids))
                    .group_by(Chunk.obj_id)
                ).all()
            )

    items = []
    for r in rows:
        chunks = chunk_counts.get(r.obj_id, 0)
        embedded = embedded_counts.get(r.obj_id, 0)
        if r.obj_id is None:
            # No objects row yet. A paper with no pdf_url is not queued for
            # anything - it is metadata we will never be able to index, and
            # listing it as "awaiting download" would be a queue that never
            # drains.
            status = "awaiting_download" if r.pdf_url else "metadata_only"
        elif r.status == "failed" and chunks == 0:
            # 'failed' with nothing chunked means the download never
            # produced a file - a different problem from a PDF that
            # downloaded and then wouldn't OCR, and a different fix.
            status = "download_failed"
        elif r.status == "chunked" and chunks and embedded >= chunks:
            # 'chunked' is where the objects table stops; whether the chunks
            # reached the index lives in chunk_embeddings, and "done" is the
            # state a reader actually wants to see.
            status = "embedded"
        else:
            status = r.status

        items.append(
            {
                "doc_id": r.id,
                "title": r.title,
                "obj_id": r.obj_id,
                "status": status,
                "attempts": r.attempts or 0,
                "chunks": chunks,
                "embedded": embedded,
            }
        )

    return sorted(
        items,
        key=lambda i: (
            STATUS_ORDER.index(i["status"]) if i["status"] in STATUS_ORDER else len(STATUS_ORDER),
            -i["doc_id"],
        ),
    )


def pipeline_status() -> dict:
    """Counts of documents/objects/chunks at each pipeline stage."""
    cfg = load()
    model_hf_name = MODELS[cfg.embedder.model]["hf_name"]

    with Session(PostgresInterface.connect()) as session:
        documents_total = session.execute(
            select(func.count()).select_from(Document)
        ).scalar_one()

        # Documents whose download hasn't finished: never attempted, or
        # attempted and still being retried. A download that gave up is
        # 'failed' and is deliberately not counted here - it is not pending,
        # nothing will move it, and leaving it in this number is what made
        # the counter look permanently stuck.
        pending_download = session.execute(
            select(func.count())
            .select_from(Document)
            .where(Document.pdf_url.is_not(None))
            .where(Document.pdf_url != "")
            .where(
                ~Document.id.in_(select(Object.doc_id))
                | Document.id.in_(
                    select(Object.doc_id).where(Object.status == "downloading")
                )
            )
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
