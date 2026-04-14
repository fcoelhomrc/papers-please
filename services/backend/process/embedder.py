import logging
import os

import numpy as np
from db.connection import PostgresInterface
from db.models import Chunk, ChunkEmbedding, EmbeddingModel
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from sentence_transformers import CrossEncoder, SentenceTransformer
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

MODELS: dict[str, dict] = {
    "bge-small": {
        "hf_name": "BAAI/bge-small-en-v1.5",
        "batch_size": 128,
        "query_prompt": "Represent this sentence for searching relevant passages: ",
        "embed_size": 384,
        "index_name": "papers-please-bge-small",
    },
    "bge-large": {
        "hf_name": "BAAI/bge-large-en-v1.5",
        "batch_size": 64,
        "query_prompt": "Represent this sentence for searching relevant passages: ",
        "embed_size": 1024,
        "index_name": "papers-please-bge-large",
    },
}

RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_id: str = RERANKER_MODEL_ID):
        self._model = CrossEncoder(model_id)

    def rerank(
        self, query: str, chunks: list[dict], top_k: int | None = None
    ) -> list[dict]:
        pairs = [(query, c["text"]) for c in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        results = [{"score": float(s), **c} for s, c in ranked]
        return results[:top_k] if top_k else results


class PdfEmbedder(PostgresInterface):
    def __init__(self, model_key: str | None = None):
        from config import load

        super().__init__()
        cfg = MODELS[model_key or load().embedder.model]
        self._cfg = cfg
        self._encoder = SentenceTransformer(cfg["hf_name"])
        self._pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    def ensure_index(self, recreate: bool = False) -> None:
        name = self._cfg["index_name"]
        if recreate and self._pc.has_index(name):
            self._pc.delete_index(name)
            logger.info(f"Dropped index {name!r}")
        if not self._pc.has_index(name):
            self._pc.create_index(
                name=name,
                vector_type="dense",
                dimension=self._cfg["embed_size"],
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="eu-west-1"),
            )
            logger.info(f"Created index {name!r}")

    def _upsert_model_record(self) -> int:
        with Session(self.engine) as session:
            session.execute(
                insert(EmbeddingModel)
                .values(
                    hf_name=self._cfg["hf_name"],
                    dims=self._cfg["embed_size"],
                    index_name=self._cfg["index_name"],
                )
                .on_conflict_do_nothing(index_elements=["hf_name"])
            )
            session.commit()
            return session.execute(
                select(EmbeddingModel.id).where(
                    EmbeddingModel.hf_name == self._cfg["hf_name"]
                )
            ).scalar_one()

    def pending(self, model_id: int) -> list[tuple[int, str, int | None]]:
        already_embedded = select(ChunkEmbedding.chunk_id).where(
            ChunkEmbedding.model_id == model_id
        )
        stmt = (
            select(Chunk.id, Chunk.chunk_text, Chunk.page_num)
            .where(Chunk.chunk_text.is_not(None))
            .where(Chunk.id.not_in(already_embedded))
        )
        with Session(self.engine) as session:
            rows = session.execute(stmt).all()
        logger.info(f"{len(rows)} chunks pending embedding (model_id={model_id})")
        return [(r.id, r.chunk_text, r.page_num) for r in rows]

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._encoder.encode(
            texts,
            batch_size=self._cfg["batch_size"],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def _upsert_vectors(self, batch: list[tuple[int, np.ndarray, int | None]]):
        index = self._pc.Index(self._cfg["index_name"])
        vectors = [
            {
                "id": str(chunk_id),
                "values": vec.tolist(),
                "metadata": {"page_num": page},
            }
            for chunk_id, vec, page in batch
        ]
        index.upsert(vectors=vectors)  # type: ignore

    def _record_embeddings(self, chunk_ids: list[int], model_id: int):
        rows = [{"chunk_id": cid, "model_id": model_id} for cid in chunk_ids]
        with Session(self.engine) as session:
            session.execute(
                insert(ChunkEmbedding).on_conflict_do_nothing(
                    index_elements=["chunk_id", "model_id"]
                ),
                rows,
            )
            session.commit()

    def execute(self, recreate_index: bool = False, max_chunks: int | None = None):
        from config import load

        self.ensure_index(recreate=recreate_index)
        model_id = self._upsert_model_record()
        limit = max_chunks if max_chunks is not None else load().embedder.max_chunks
        pending = self.pending(model_id)[:limit]
        if not pending:
            logger.info("Nothing to embed")
            return

        batch_size = self._cfg["batch_size"]
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            chunk_ids = [r[0] for r in batch]
            texts = [r[1] for r in batch]
            pages = [r[2] for r in batch]

            vecs = self._embed(texts)
            self._upsert_vectors(list(zip(chunk_ids, vecs, pages)))
            self._record_embeddings(chunk_ids, model_id)
            logger.info(f"Embedded {i + len(batch)}/{len(pending)} chunks")

        logger.info(f"Done — {len(pending)} chunks embedded")
