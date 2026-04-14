from contextlib import asynccontextmanager
from pathlib import Path

import log
from config import load
from db.models import Document, Object
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from ingest.fetcher import SemanticScholarFetcher
from process.embedder import MODELS, Reranker
from schemas import DocumentOut, FetchRequest, SearchResponse
from search import SearchEngine
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.orm import Session

log.setup()

_engine: SearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    cfg = load()
    model_key = cfg.embedder.model
    encoder = SentenceTransformer(MODELS[model_key]["hf_name"], device=cfg.devices.embedder)
    reranker = Reranker(device=cfg.devices.reranker)
    _engine = SearchEngine(encoder=encoder, reranker=reranker, model_key=model_key)
    yield


app = FastAPI(title="Papers Please", lifespan=lifespan)


def get_engine() -> SearchEngine:
    assert _engine is not None
    return _engine


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
