from contextlib import asynccontextmanager

import log
from config import load
from db.models import Document
from fastapi import Depends, FastAPI, Query
from process.embedder import MODELS, Reranker
from schemas import DocumentOut, SearchResponse
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
    encoder = SentenceTransformer(MODELS[model_key]["hf_name"])
    reranker = Reranker()
    _engine = SearchEngine(encoder=encoder, reranker=reranker, model_key=model_key)
    yield


app = FastAPI(title="Papers Please", lifespan=lifespan)


def get_engine() -> SearchEngine:
    assert _engine is not None
    return _engine


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search", response_model=SearchResponse)
def search(
    q: str,
    top_k: int = Query(default=10, ge=1, le=50),
    rerank: bool = False,
    rerank_top_k: int = Query(default=5, ge=1, le=20),
    engine: SearchEngine = Depends(get_engine),
):
    return engine.search(q, top_k=top_k, rerank=rerank, rerank_top_k=rerank_top_k)


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
