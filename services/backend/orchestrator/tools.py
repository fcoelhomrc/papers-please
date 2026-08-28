"""LangChain tools for the orchestrator agent.

Deliberately small. The ingest pipeline (download -> chunk -> embed) is
deterministic: given the DB state there's exactly one correct action, so
it's handled by the stages/*.py workers running on a timer, not by an LLM
deciding whether to call a tool. An agent has no business being a queue
manager for that - it adds latency/cost/a new way to silently skip a step,
for a decision that was never actually being made.

What's left is where an LLM's judgment is real: what to search for (fetch),
what/how to query for retrieval (search_chunks/get_document), and get_status
as context for a fetch decision (e.g. "we already have papers on this",
don't refetch) - not a router between pipeline stages.
"""
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from config import load
from db.connection import PostgresInterface
from db.models import Chunk, ChunkEmbedding, Document, Object
from ingest.fetcher import SemanticScholarFetcher
from langchain_core.tools import tool
from process.embedder import MODELS
from search import get_search_engine


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


@tool
def search_chunks(query: str, top_k: int = 5, rerank: bool = True) -> list[dict]:
    """Search the paper library for chunks relevant to a question.

    Returns chunk text with its doc_id, title, and page - cite doc_id/page
    when answering so the user can find the source. Use get_document for
    more of a paper's context (e.g. its abstract) once you've found it here.
    """
    response = get_search_engine().search(query, top_k=top_k, rerank=rerank, rerank_top_k=top_k)
    return [
        {
            "doc_id": r.doc_id,
            "title": r.title,
            "authors": r.authors,
            "year": r.year,
            "page_num": r.page_num,
            "text": r.text,
            "score": r.score,
        }
        for r in response.results
    ]


@tool
def get_document(doc_id: int) -> dict:
    """Look up a paper's metadata (title, authors, year, abstract) by doc_id."""
    with Session(PostgresInterface.connect()) as session:
        doc = session.get(Document, doc_id)
    if doc is None:
        return {"error": f"no document with doc_id={doc_id}"}
    return {
        "doc_id": doc.id,
        "title": doc.title,
        "authors": doc.authors,
        "venue": doc.venue,
        "year": doc.year,
        "abstract": doc.abstract,
    }
