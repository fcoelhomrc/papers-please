from contextlib import asynccontextmanager
from pathlib import Path

import log
from config import load
from db.models import Chunk, ChunkEmbedding, Document, Object
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from ingest.fetcher import SemanticScholarFetcher
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from orchestrator.graph import build_agent
from orchestrator.llm import make_llm
from schemas import (
    ChatRequest,
    ChatResponse,
    DocumentOut,
    FetchRequest,
    SearchResponse,
    StatusResponse,
)
from search import SearchEngine, get_search_engine, keyword_search
from sqlalchemy import exists, select
from sqlalchemy.orm import Session
from status import pipeline_status

log.setup()

_agent = None  # built lazily - needs ANTHROPIC_API_KEY, shouldn't block startup without it


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_search_engine()  # pre-warm at startup rather than on the first request
    yield


app = FastAPI(title="Papers Please", lifespan=lifespan)


def get_engine() -> SearchEngine:
    return get_search_engine()


def get_agent():
    global _agent
    if _agent is None:
        try:
            # MemorySaver gives the agent conversation memory across turns,
            # keyed by thread_id - in-process only, resets on backend
            # restart, which is fine for a single-instance dev deployment.
            _agent = build_agent(make_llm(load()), checkpointer=MemorySaver())
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"orchestrator agent unavailable: {e}",
            ) from e
    return _agent


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/fetch")
def fetch(req: FetchRequest):
    total = SemanticScholarFetcher().fetch(
        query=req.query,
        venue=req.venue,
        year=req.year,
        max_papers=req.max_papers,
    )
    return {"fetched": total}


@app.get("/status", response_model=StatusResponse)
def status():
    return pipeline_status()


@app.post("/agent/chat", response_model=ChatResponse)
def agent_chat(req: ChatRequest, agent=Depends(get_agent)):
    config = {"configurable": {"thread_id": req.thread_id}}

    # With memory, result["messages"] holds the *whole* conversation - need
    # to know how many messages existed before this turn so tool_calls only
    # reflects what just happened, not every past turn too.
    prior_state = agent.get_state(config)
    prior_count = len(prior_state.values.get("messages", [])) if prior_state.values else 0

    result = agent.invoke({"messages": [HumanMessage(req.message)]}, config=config)
    new_messages = result["messages"][prior_count:]

    tool_calls = [
        call["name"]
        for m in new_messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]
    return ChatResponse(reply=result["messages"][-1].content, tool_calls=tool_calls)


@app.get("/search", response_model=SearchResponse)
def search(
    q: str,
    top_k: int = Query(default=10, ge=1, le=50),
    rerank: bool = False,
    rerank_top_k: int = Query(default=5, ge=1, le=20),
    engine: SearchEngine = Depends(get_engine),
):
    return engine.search(q, top_k=top_k, rerank=rerank, rerank_top_k=rerank_top_k)


@app.get("/search/keyword", response_model=SearchResponse)
def search_keyword(q: str, top_k: int = Query(default=10, ge=1, le=50)):
    return keyword_search(q, top_k=top_k)


def _pdf_and_processed_ids(session: Session, doc_ids: list[int]) -> tuple[set[int], set[int]]:
    """Which of these doc_ids have a downloaded PDF (an objects row) and
    which are fully processed (at least one chunk with an embedding) -
    batched rather than N+1, computed separately from the main query since
    expressing both as correlated subqueries in one ORM select got unreadable
    fast for little benefit at this table size."""
    if not doc_ids:
        return set(), set()
    pdf_ids = set(
        session.execute(
            select(Object.doc_id).where(Object.doc_id.in_(doc_ids)).distinct()
        ).scalars().all()
    )
    processed_ids = set(
        session.execute(
            select(Object.doc_id)
            .select_from(Chunk)
            .join(Object, Chunk.obj_id == Object.id)
            .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
            .where(Object.doc_id.in_(doc_ids))
            .distinct()
        ).scalars().all()
    )
    return pdf_ids, processed_ids


def _document_out(doc: Document, pdf_ids: set[int], processed_ids: set[int]) -> DocumentOut:
    return DocumentOut(
        id=doc.id,
        source_id=doc.source_id,
        title=doc.title,
        authors=doc.authors,
        venue=doc.venue,
        year=doc.year,
        abstract=doc.abstract,
        has_pdf=doc.id in pdf_ids,
        processed=doc.id in processed_ids,
    )


@app.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, engine: SearchEngine = Depends(get_engine)):
    with Session(engine.engine) as session:
        doc = session.get(Document, doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        pdf_ids, processed_ids = _pdf_and_processed_ids(session, [doc_id])
    return _document_out(doc, pdf_ids, processed_ids)


@app.get("/documents/{doc_id}/pdf")
def get_pdf(
    doc_id: int,
    download: bool = Query(default=False),
    engine: SearchEngine = Depends(get_engine),
):
    with Session(engine.engine) as session:
        path = session.execute(
            select(Object.path).where(Object.doc_id == doc_id)
        ).scalar_one_or_none()
    if path is None:
        raise HTTPException(status_code=404, detail="PDF not found")
    full_path = Path(load().storage.root) / path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="PDF file missing from storage")
    # FileResponse defaults to Content-Disposition: attachment, which forces
    # a download even inside an <iframe> - "inline" lets the browser render
    # it in place for the preview dialog; ?download=true opts into the
    # explicit Download button's behavior instead.
    return FileResponse(
        full_path,
        media_type="application/pdf",
        filename=path,
        content_disposition_type="attachment" if download else "inline",
    )


SORT_OPTIONS = {
    "newest": Document.created_at.desc(),
    "oldest": Document.created_at.asc(),
    "title": Document.title.asc(),
    "year": Document.year.desc().nulls_last(),
}


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, le=100),
    q: str | None = Query(default=None, description="Filter by title (substring, case-insensitive)"),
    only_available: bool = Query(default=False, description="Only papers with a downloaded PDF"),
    only_processed: bool = Query(default=False, description="Only papers fully searchable (chunked + embedded)"),
    sort: str = Query(default="newest", pattern="^(newest|oldest|title|year)$"),
    engine: SearchEngine = Depends(get_engine),
):
    with Session(engine.engine) as session:
        stmt = select(Document)
        if q:
            stmt = stmt.where(Document.title.ilike(f"%{q}%"))
        if only_available:
            stmt = stmt.where(exists(select(1).where(Object.doc_id == Document.id)))
        if only_processed:
            stmt = stmt.where(
                exists(
                    select(1)
                    .select_from(Chunk)
                    .join(Object, Chunk.obj_id == Object.id)
                    .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
                    .where(Object.doc_id == Document.id)
                )
            )
        stmt = stmt.order_by(SORT_OPTIONS[sort]).offset(offset).limit(limit)

        docs = session.execute(stmt).scalars().all()
        pdf_ids, processed_ids = _pdf_and_processed_ids(session, [d.id for d in docs])
    return [_document_out(d, pdf_ids, processed_ids) for d in docs]
