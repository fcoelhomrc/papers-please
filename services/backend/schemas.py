from pydantic import BaseModel, ConfigDict, Field


class FetchRequest(BaseModel):
    query: str = ""
    venue: str | None = None
    year: str | None = None
    max_papers: int = Field(default=500, ge=1, le=5000)


class ChunkResult(BaseModel):
    chunk_id: int
    doc_id: int
    title: str
    authors: list[str] | None
    year: int | None
    page_num: int | None
    pdf_path: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    model: str
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
