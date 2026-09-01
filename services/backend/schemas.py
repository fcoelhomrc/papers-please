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
    has_pdf: bool = False  # a PDF was downloaded and registered (objects row exists)
    processed: bool = False  # fully chunked + embedded, i.e. searchable
