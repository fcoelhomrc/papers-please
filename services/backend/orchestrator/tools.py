"""LangChain tools wrapping the existing pipeline stages.

Each function below is decorated with @tool (from langchain_core.tools),
which turns it into a langchain "Tool" object: the docstring becomes the
description an LLM sees when deciding whether to call it, and the type
hints become a JSON schema for its arguments. This is plumbing only - the
actual agent loop that decides *when* to call these lands in subtask 4
(LangGraph's create_react_agent). For now these are just callable directly,
same as any Python function, which is how the unit tests exercise them.

None of the underlying .execute() methods (PdfFetcher/PdfChunker/PdfEmbedder)
return a count - they log and return None. So each wrapper snapshots
len(pending()) *before* calling execute() to give the LLM something
concrete to report, rather than a meaningless "done".
"""
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from config import load
from db.connection import PostgresInterface
from db.models import Chunk, ChunkEmbedding, Document, Object
from ingest.fetcher import PdfFetcher, SemanticScholarFetcher
from langchain_core.tools import tool
from process.chunker import PdfChunker
from process.embedder import MODELS, PdfEmbedder


@tool
def fetch_papers(query: str, max_papers: int = 100) -> str:
    """Fetch new paper metadata from Semantic Scholar for a search query."""
    n = SemanticScholarFetcher().fetch(query=query, max_papers=max_papers)
    return f"fetched {n} papers"


@tool
def download_pending(limit: int = 20) -> str:
    """Download PDFs for documents that have a pdf_url but no downloaded object yet."""
    cfg = load()
    fetcher = PdfFetcher(max_workers=cfg.stages.download.workers)
    n = len(fetcher.pending()[:limit])
    fetcher.execute(limit=limit)
    return f"attempted {n} downloads (limit={limit})"


@tool
def chunk_pending(limit: int = 10) -> str:
    """OCR + chunk downloaded PDFs that are still in 'pending' status."""
    chunker = PdfChunker()
    n = len(chunker.pending()[:limit])
    chunker.execute(limit=limit)
    return f"attempted {n} objects to chunk (limit={limit})"


@tool
def embed_pending(limit: int = 500) -> str:
    """Embed chunks that don't have a vector yet and upsert them to Pinecone."""
    embedder = PdfEmbedder()
    return _run_embed(embedder, limit)


def _run_embed(embedder: PdfEmbedder, limit: int) -> str:
    # PdfEmbedder needs a model_id to know what's "pending", which only
    # exists once _upsert_model_record() has run - so we can't snapshot a
    # count up front the way the other tools do. Just report the ceiling.
    embedder.execute(max_chunks=limit)
    return f"embedded up to {limit} pending chunks"


@tool
def get_status() -> dict:
    """Return counts of documents/objects/chunks at each pipeline stage.

    This is what lets the orchestrator agent decide which stage tool to
    call next instead of guessing - e.g. if pending_download is high but
    objects_by_status.pending is 0, download is the bottleneck.
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
        # counts as pending. Good enough for routing; doesn't distinguish
        # "never embedded" from "embedded with a different model".
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
