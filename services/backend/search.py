import os

from pinecone.grpc import PineconeGRPC as Pinecone
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.connection import PostgresInterface
from db.models import Chunk, Document, Object
from schemas import ChunkResult, SearchResponse
from process.embedder import MODELS, Reranker


class SearchEngine(PostgresInterface):
    def __init__(self, model_key: str = "bge-small"):
        super().__init__()
        cfg = MODELS[model_key]
        self._cfg = cfg
        self._model_key = model_key
        self._encoder = SentenceTransformer(cfg["hf_name"])
        self._pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self._reranker = Reranker()

    def _embed_query(self, query: str) -> list[float]:
        vec = self._encoder.encode(
            query,
            prompt=self._cfg["query_prompt"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = False,
        rerank_top_k: int = 5,
    ) -> SearchResponse:
        vec = self._embed_query(query)

        index = self._pc.Index(self._cfg["index_name"])
        response = index.query(vector=vec, top_k=top_k, include_metadata=True)
        matches = response["matches"]

        if not matches:
            return SearchResponse(query=query, model=self._model_key, reranked=False, results=[])

        chunk_ids = [int(m["id"]) for m in matches]
        scores = {int(m["id"]): m["score"] for m in matches}

        stmt = (
            select(Chunk.id, Chunk.chunk_text, Chunk.page_num, Document.title, Document.authors, Document.year)
            .join(Object, Chunk.obj_id == Object.id)
            .join(Document, Object.doc_id == Document.id)
            .where(Chunk.id.in_(chunk_ids))
        )
        with Session(self.engine) as session:
            rows = session.execute(stmt).all()

        chunks = [
            {
                "chunk_id": r.id,
                "text": r.chunk_text,
                "page_num": r.page_num,
                "title": r.title,
                "authors": r.authors,
                "year": r.year,
                "score": scores[r.id],
            }
            for r in rows
        ]
        chunks.sort(key=lambda c: c["score"], reverse=True)

        if rerank:
            chunks = self._reranker.rerank(query, chunks, top_k=rerank_top_k)

        return SearchResponse(
            query=query,
            model=self._model_key,
            reranked=rerank,
            results=[ChunkResult(**c) for c in chunks],
        )
