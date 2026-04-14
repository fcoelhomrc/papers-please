from pydantic import BaseModel, ConfigDict


class ChunkResult(BaseModel):
    chunk_id: int
    title: str
    authors: list[str] | None
    year: int | None
    page_num: int | None
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
