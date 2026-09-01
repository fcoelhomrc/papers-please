from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FetchRequest(BaseModel):
    query: str = ""
    venue: str | None = None
    year: str | None = None
    max_papers: int = Field(default=500, ge=1, le=5000)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"  # groups turns into one conversation for the agent's memory


class Evidence(BaseModel):
    """One chunk the agent retrieved, as a citation card.

    Carries no chunk text: it is already in the model's answer, and shipping
    it again would multiply the response size for a card that shows a title
    and a page number. The UI fetches the passage from /search or the PDF
    when someone actually opens it.
    """

    doc_id: int | None = None
    chunk_id: int | None = None
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    page_num: int | None = None
    score: float | None = None


class TraceStep(BaseModel):
    """One tool call, for the "what did it do" strip under an answer."""

    tool: str
    args: dict = {}
    summary: str = ""
    ok: bool = True


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[str] = []  # names of tools the agent invoked, for transparency
    # The chunks behind the answer's [doc N, pX] citations, so the UI can turn
    # unlinked prose into something openable and checkable.
    evidence: list[Evidence] = []
    trace: list[TraceStep] = []


class FeedbackRequest(BaseModel):
    """A thumbs-up/down on one result or citation.

    `query` is required even for a citation: a judgement with no question
    attached cannot become an eval row, which is the entire reason for
    collecting these.
    """

    kind: str = Field(default="search", pattern="^(search|citation)$")
    query: str = Field(min_length=1)
    doc_id: int | None = None
    chunk_id: int | None = None
    verdict: str = Field(pattern="^(up|down)$")
    note: str | None = None


class FeedbackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    kind: str
    query: str
    doc_id: int | None
    chunk_id: int | None
    verdict: str
    note: str | None
    created_at: datetime


class QueueItem(BaseModel):
    """One paper's position in the pipeline, for the Queue page.

    The page showed aggregate counts only, so "what is it doing right now"
    was unanswerable and a stuck document was indistinguishable from a busy
    pipeline.
    """

    doc_id: int
    title: str
    obj_id: int | None = None
    status: str  # 'awaiting_download' | 'pending' | 'chunked' | 'failed' | 'dead' | 'embedded'
    attempts: int = 0
    chunks: int = 0
    embedded: int = 0


class WorkerStatus(BaseModel):
    """One pipeline worker as the container runtime sees it.

    `state` is the runtime's own word (`running`, `exited`, `created`, ...)
    plus `missing` for a service with no container, and `unknown` when the
    runtime itself couldn't be reached.
    """

    service: str
    container: str | None = None
    state: str
    status: str = ""
    exit_code: int | None = None


class WorkersResponse(BaseModel):
    workers: list[WorkerStatus] = []
    # Set when the container runtime is unreachable - normal when the
    # backend runs outside compose. The UI shows "unknown" rather than
    # treating it as workers being down, which would be a worse lie than
    # saying nothing.
    unavailable: str | None = None


class StatusResponse(BaseModel):
    documents_total: int
    pending_download: int
    objects_by_status: dict[str, int]
    chunks_pending_embed: int
    embed_model: str


class ChunkResult(BaseModel):
    chunk_id: int
    chunk_index: int
    doc_id: int
    title: str
    authors: list[str] | None
    year: int | None
    page_num: int | None
    pdf_path: str
    # `text` is the chunk that actually matched - what the score refers to and
    # what the UI highlights. `context` is that chunk plus its neighbours, for
    # a reader (human or LLM) who needs the sentence that ran over the chunk
    # boundary. Kept separate rather than widening `text` in place: conflating
    # them would make `score` look like it applied to the whole window, and
    # the UI would lose the ability to show what actually matched.
    text: str
    context: str | None = None
    score: float
    # Whether this chunk's paper has a PDF on disk. Without it the result
    # card offers a Preview whose only possible outcome is a 404 - which is
    # every result in a library seeded from eval fixtures, since those have
    # chunk text but no file behind it.
    has_pdf: bool = False
    # Which retriever(s) surfaced this chunk. Only fusion knows - afterwards a
    # chunk both retrievers agreed on looks identical to one either found
    # alone, and agreement is a different kind of confidence from one
    # retriever being very sure.
    sources: list[str] = []


class DocumentChunk(BaseModel):
    """A chunk as listed on a paper's own page, in document order."""

    chunk_id: int
    chunk_index: int
    page_num: int | None
    text: str


class SearchResponse(BaseModel):
    query: str
    model: str
    mode: str = "semantic"  # which retrieval mode produced these results
    reranked: bool
    results: list[ChunkResult]


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source_id: str
    title: str
    authors: list[str] | None
    venue: str | None
    year: int | None
    abstract: str | None
    has_pdf: bool = False  # the PDF is readable on disk, not merely recorded
    processed: bool = False  # fully chunked + embedded, i.e. searchable
