from pydantic import BaseModel, ConfigDict, Field


class FetchRequest(BaseModel):
    query: str = ""
    venue: str | None = None
    year: str | None = None
    max_papers: int = Field(default=500, ge=1, le=5000)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"  # groups turns into one conversation for the agent's memory


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[str] = []  # names of tools the agent invoked, for transparency


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
