import json
from contextlib import asynccontextmanager
from pathlib import Path

import log
from config import load
from db.models import Chunk, ChunkEmbedding, Document, Feedback, Object
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from ingest.fetcher import SemanticScholarFetcher
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from orchestrator.graph import MAX_AGENT_RECURSION, build_agent
from orchestrator.llm import make_agent_parts
from orchestrator.evidence import extract_evidence, extract_trace
from fastapi.responses import PlainTextResponse
from schemas import (
    ChatRequest,
    ChatResponse,
    DocumentChunk,
    DocumentOut,
    FeedbackOut,
    FeedbackRequest,
    FetchRequest,
    QueueItem,
    SearchResponse,
    StatusResponse,
    WorkersResponse,
    WorkerStatus,
)
from search import SearchEngine, get_search_engine, keyword_search
from sqlalchemy import exists, select
from sqlalchemy.orm import Session
from status import pipeline_status, queue_items

log.setup()

_agent = None  # built lazily - needs ANTHROPIC_API_KEY, shouldn't block startup without it


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Deferred to lifespan (not module import time) same as get_search_engine
    # below - plain TestClient(app) instantiation skips lifespan entirely, so
    # tests never attempt a real OTel connection. Learned this the hard way:
    # calling it at import time hung every test that imports api.py trying to
    # reach an OTel collector nothing was running.
    from observability import setup_observability

    setup_observability("papers-please-backend")
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
            llm, tools = make_agent_parts(load())
            _agent = build_agent(llm, checkpointer=MemorySaver(), tools=tools)
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


@app.get("/queue", response_model=list[QueueItem])
def queue(limit: int = Query(default=50, ge=1, le=200)):
    """What is in the pipeline and where each paper has got to."""
    return queue_items(limit=limit)


@app.get("/workers", response_model=WorkersResponse)
def workers():
    """Whether the pipeline workers are actually running.

    The Queue page can show a full backlog and a stopped worker at the same
    time and, before this, looked identical in both cases - which is how six
    stuck downloads got reported as a broken pipeline when the download
    worker simply wasn't up.
    """
    import containers

    try:
        return WorkersResponse(
            workers=[WorkerStatus(**w) for w in containers.list_workers()]
        )
    except Exception as e:
        # Never a 500: running the backend outside compose, or without the
        # podman socket enabled, is a normal setup and not an error.
        return WorkersResponse(unavailable=str(e))


@app.get("/workers/{service}/logs", response_class=PlainTextResponse)
def worker_logs(service: str, tail: int = Query(default=200, ge=1, le=2000)):
    """Recent output from one worker, so a failure is readable without a
    terminal. Read-only - there is no start/stop here."""
    import containers

    try:
        return containers.worker_logs(service, tail=tail)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"container runtime unavailable: {e}") from e


@app.post("/agent/chat", response_model=ChatResponse)
def agent_chat(req: ChatRequest, agent=Depends(get_agent)):
    config = {"configurable": {"thread_id": req.thread_id}, "recursion_limit": MAX_AGENT_RECURSION}

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
    # Extracted from this turn's messages only, same as tool_calls - with
    # memory, result["messages"] holds the whole conversation, and citing
    # three turns' worth of evidence under one answer would attribute
    # sources to claims that never used them.
    return ChatResponse(
        reply=result["messages"][-1].content,
        tool_calls=tool_calls,
        evidence=extract_evidence(new_messages),
        trace=extract_trace(new_messages),
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/agent/chat/stream")
async def agent_chat_stream(req: ChatRequest, agent=Depends(get_agent)):
    """The same turn as /agent/chat, reported as it happens.

    Non-streaming, the panel shows a bouncing ellipsis for however long the
    agent takes and then everything lands at once - the tool calls are only
    visible after they no longer matter. Here each graph node emits a `step`
    as it completes, so a search announces itself while it is running.

    stream_mode="updates" rather than astream_events: node updates are a
    small documented shape (node name -> the messages it added), where the
    event stream is a much larger surface to depend on for the same
    information. Token-level streaming is deliberately not used - the reply
    is short and the interesting events here are the tool calls.

    /agent/chat is unchanged and still the simpler thing to call.
    """
    config = {
        "configurable": {"thread_id": req.thread_id},
        "recursion_limit": MAX_AGENT_RECURSION,
    }
    prior_state = agent.get_state(config)
    prior_count = len(prior_state.values.get("messages", [])) if prior_state.values else 0

    async def events():
        try:
            async for update in agent.astream(
                {"messages": [HumanMessage(req.message)]}, config=config, stream_mode="updates"
            ):
                for node, payload in update.items():
                    for message in (payload or {}).get("messages", []) or []:
                        if isinstance(message, AIMessage) and message.tool_calls:
                            for call in message.tool_calls:
                                yield _sse(
                                    "step",
                                    {"kind": "tool_call", "tool": call["name"], "args": call.get("args") or {}},
                                )
                        elif isinstance(message, ToolMessage):
                            yield _sse("step", {"kind": "tool_result", "tool": message.name})
                        elif isinstance(message, AIMessage) and node != "tools":
                            yield _sse("step", {"kind": "thinking"})
        except Exception as e:
            # The stream is already open, so an HTTP error code is no longer
            # available - the client has to learn about this in-band or it
            # waits forever on a `done` that never comes.
            yield _sse("error", {"detail": str(e)})
            return

        # Re-read the final state rather than accumulating as we go: the
        # checkpointer holds the authoritative message list, and rebuilding
        # it from stream fragments is a second place for the two to disagree.
        state = agent.get_state(config)
        messages = state.values.get("messages", [])
        new_messages = messages[prior_count:]
        yield _sse(
            "done",
            ChatResponse(
                reply=messages[-1].content if messages else "",
                tool_calls=[
                    call["name"]
                    for m in new_messages
                    if isinstance(m, AIMessage)
                    for call in m.tool_calls
                ],
                evidence=extract_evidence(new_messages),
                trace=extract_trace(new_messages),
            ).model_dump(),
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Without this, nginx (services/frontend/nginx.conf proxies /api)
        # buffers the whole response and the stream arrives as one lump -
        # which looks exactly like the non-streaming endpoint and makes the
        # bug hard to spot.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/feedback", response_model=FeedbackOut)
def create_feedback(req: FeedbackRequest, engine: SearchEngine = Depends(get_engine)):
    """Record a relevance judgement.

    Every search and every cited answer is a labelling opportunity, and the
    eval set is otherwise grown by hand (eval/fixtures.py). Stored rather
    than sent to Phoenix: Phoenix only sees LangChain spans, so /search -
    which never touches LangChain - produces nothing to annotate, and its
    volume is disposable dev state. Labels have to outlive that.
    """
    row = Feedback(**req.model_dump())
    with Session(engine.engine) as session:
        session.add(row)
        session.commit()
        session.refresh(row)
        return FeedbackOut.model_validate(row)


@app.get("/feedback", response_model=list[FeedbackOut])
def list_feedback(
    verdict: str | None = Query(default=None, pattern="^(up|down)$"),
    limit: int = Query(default=200, ge=1, le=1000),
    engine: SearchEngine = Depends(get_engine),
):
    with Session(engine.engine) as session:
        stmt = select(Feedback).order_by(Feedback.created_at.desc()).limit(limit)
        if verdict:
            stmt = stmt.where(Feedback.verdict == verdict)
        return [FeedbackOut.model_validate(r) for r in session.execute(stmt).scalars().all()]


@app.get("/search", response_model=SearchResponse)
def search(
    q: str,
    top_k: int = Query(default=10, ge=1, le=50),
    rerank: bool = False,
    rerank_top_k: int = Query(default=5, ge=1, le=20),
    # None -> whatever config.search.mode says, so the default is one place
    mode: str | None = Query(default=None, pattern="^(semantic|keyword|hybrid)$"),
    engine: SearchEngine = Depends(get_engine),
):
    return engine.search(q, top_k=top_k, rerank=rerank, rerank_top_k=rerank_top_k, mode=mode)


@app.get("/search/keyword", response_model=SearchResponse)
def search_keyword(q: str, top_k: int = Query(default=10, ge=1, le=50)):
    return keyword_search(q, top_k=top_k)


def _pdf_and_processed_ids(session: Session, doc_ids: list[int]) -> tuple[set[int], set[int]]:
    """Which of these doc_ids have a readable PDF on disk, and which are
    fully processed (at least one chunk with an embedding) - batched rather
    than N+1, computed separately from the main query since expressing both
    as correlated subqueries in one ORM select got unreadable fast for little
    benefit at this table size.

    `has_pdf` used to mean "an objects row exists", which is not the same
    claim and is the one the UI acts on: it decides whether to offer a
    Preview button. Normally the two agree, because the download stage
    writes the file before inserting the row - they diverge exactly where it
    matters. eval/seed.py inserts an objects row with a synthetic path for
    fixtures that have no PDF at all, and a wiped storage volume or a
    misconfigured storage.root does the same thing to real papers. In every
    one of those cases the old answer produced a button whose only possible
    outcome was a 404.

    One stat per object on the page, bounded by the endpoint's limit.
    """
    if not doc_ids:
        return set(), set()
    root = Path(load().storage.root)
    pdf_ids = {
        r.doc_id
        for r in session.execute(
            select(Object.doc_id, Object.path).where(Object.doc_id.in_(doc_ids))
        ).all()
        if r.path and (root / r.path).is_file()
    }
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


@app.get("/documents/{doc_id}/chunks", response_model=list[DocumentChunk])
def get_document_chunks(
    doc_id: int,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    engine: SearchEngine = Depends(get_engine),
):
    """A paper's chunks in document order - what the index actually holds for
    it, which is otherwise only observable by searching for a phrase you
    already know is in there.

    Paginated: a long paper at 256-token chunks runs to a few hundred rows,
    and the page renders them lazily rather than shipping the whole PDF's
    text on first paint.
    """
    with Session(engine.engine) as session:
        rows = session.execute(
            select(Chunk.id, Chunk.chunk_index, Chunk.page_num, Chunk.chunk_text)
            .join(Object, Chunk.obj_id == Object.id)
            .where(Object.doc_id == doc_id)
            .order_by(Chunk.chunk_index)
            .offset(offset)
            .limit(limit)
        ).all()
    return [
        DocumentChunk(
            chunk_id=r.id, chunk_index=r.chunk_index, page_num=r.page_num, text=r.chunk_text or ""
        )
        for r in rows
    ]


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
            # Not merely "an objects row exists": a row is created when a
            # download is first attempted, so that would list papers whose
            # download is still running or has given up entirely.
            stmt = stmt.where(
                exists(
                    select(1).where(
                        (Object.doc_id == Document.id)
                        & (Object.status.not_in(("downloading", "failed")))
                    )
                )
            )
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
