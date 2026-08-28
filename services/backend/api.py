from contextlib import asynccontextmanager
from pathlib import Path

import log
from config import load
from db.models import Document, Object
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from ingest.fetcher import SemanticScholarFetcher
from langchain_core.messages import AIMessage, HumanMessage
from orchestrator.graph import build_agent
from orchestrator.llm import make_llm
from schemas import ChatRequest, ChatResponse, DocumentOut, FetchRequest, SearchResponse
from search import SearchEngine, get_search_engine
from sqlalchemy import select
from sqlalchemy.orm import Session

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
            _agent = build_agent(make_llm(load()))
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


@app.post("/agent/chat", response_model=ChatResponse)
def agent_chat(req: ChatRequest, agent=Depends(get_agent)):
    result = agent.invoke({"messages": [HumanMessage(req.message)]})
    messages = result["messages"]
    tool_calls = [
        call["name"]
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]
    return ChatResponse(reply=messages[-1].content, tool_calls=tool_calls)


@app.get("/search", response_model=SearchResponse)
def search(
    q: str,
    top_k: int = Query(default=10, ge=1, le=50),
    rerank: bool = False,
    rerank_top_k: int = Query(default=5, ge=1, le=20),
    engine: SearchEngine = Depends(get_engine),
):
    return engine.search(q, top_k=top_k, rerank=rerank, rerank_top_k=rerank_top_k)


@app.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, engine: SearchEngine = Depends(get_engine)):
    with Session(engine.engine) as session:
        doc = session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentOut.model_validate(doc)


@app.get("/documents/{doc_id}/pdf")
def get_pdf(doc_id: int, engine: SearchEngine = Depends(get_engine)):
    with Session(engine.engine) as session:
        path = session.execute(
            select(Object.path).where(Object.doc_id == doc_id)
        ).scalar_one_or_none()
    if path is None:
        raise HTTPException(status_code=404, detail="PDF not found")
    full_path = Path(load().storage.root) / path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="PDF file missing from storage")
    return FileResponse(full_path, media_type="application/pdf", filename=path)


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, le=100),
    engine: SearchEngine = Depends(get_engine),
):
    with Session(engine.engine) as session:
        docs = (
            session.execute(select(Document).offset(offset).limit(limit))
            .scalars()
            .all()
        )
    return [DocumentOut.model_validate(d) for d in docs]
