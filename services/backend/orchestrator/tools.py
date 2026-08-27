"""LangChain tools for the orchestrator agent.

Deliberately small. The ingest pipeline (download -> chunk -> embed) is
deterministic: given the DB state there's exactly one correct action, so
it's handled by the stages/*.py workers running on a timer, not by an LLM
deciding whether to call a tool. An agent has no business being a queue
manager for that - it adds latency/cost/a new way to silently skip a step,
for a decision that was never actually being made.

What's left is where an LLM's judgment is real: what to search for (fetch)
and, later, what/how to query for retrieval. get_status stays because it's
useful context for a fetch decision (e.g. "we already have papers on this",
don't refetch), not because it routes between pipeline stages anymore.
"""
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from config import load
from db.connection import PostgresInterface
from db.models import Chunk, ChunkEmbedding, Document, Object
from ingest.fetcher import SemanticScholarFetcher
from langchain_core.tools import tool
from process.embedder import MODELS


@tool
def fetch_papers(query: str, max_papers: int = 100) -> str:
    """Fetch new paper metadata from Semantic Scholar for a search query."""
    n = SemanticScholarFetcher().fetch(query=query, max_papers=max_papers)
    return f"fetched {n} papers"


@tool
def get_status() -> dict:
    """Return counts of documents/objects/chunks at each pipeline stage.

    Context for fetch decisions - e.g. how much we already have on a topic
    before fetching more. Not a router for the ingest pipeline, which runs
    on its own deterministic schedule (stages/*.py).
    """
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
