import logging
import os

import numpy as np
import torch
from db.connection import PostgresInterface
from db.models import Chunk, ChunkEmbedding, Document, EmbeddingModel, Object
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
    # Each family prescribes its own query prefix, and they are not
    # interchangeable. A model asked for with the wrong prefix - or with none
    # when it expects one - retrieves noticeably worse, and the failure looks
    # exactly like the model being bad rather than like it being misconfigured.
    # These come from each model's own card, not from a house style.
    "arctic-m-v2": {
        "hf_name": "Snowflake/snowflake-arctic-embed-m-v2.0",
        "batch_size": 64,
        "query_prompt": "query: ",
        "embed_size": 768,
        "index_name": "papers-please-arctic-m-v2",
        "trust_remote_code": True,
        # The xformers in this image was built for torch 2.10 / py3.10 against
        # the 2.11 / py3.14 actually installed, so its CUDA extensions do not
        # load and the memory-efficient path ends up building an attention bias
        # on a different device than the query. sdpa is a fallback the same
        # remote code already implements, so nothing here depends on xformers.
        "config_kwargs": {
            "use_memory_efficient_attention": False,
            "unpad_inputs": False,
        },
        "repair_gte_buffers": True,
    },
    "qwen3-0.6b": {
        "hf_name": "Qwen/Qwen3-Embedding-0.6B",
        "batch_size": 32,
        # Qwen3 conditions on an instruction rather than a bare tag; the card's
        # own retrieval template is "Instruct: <task>\nQuery: <query>".
        "query_prompt": (
            "Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query\nQuery: "
        ),
        "embed_size": 1024,
        "index_name": "papers-please-qwen3-06b",
    },
}


def _repair_gte_buffers(model: SentenceTransformer) -> None:
    """Recompute the buffers transformers leaves as uninitialised memory.

    transformers 5 materialises only what the checkpoint contains, and the GTE
    remote code this family ships keeps `position_ids` and its rotary
    `inv_freq` / `cos_cached` / `sin_cached` as non-persistent buffers — absent
    from the checkpoint, so they survive loading as whatever was on the heap.
    The symptoms vary with batch shape and device and none of them names the
    cause: a CUDA device-side assert, an IndexError indexing rope by a
    nonsense position, or embeddings that are silently all NaN.

    The values come from the model's own __init__, re-run against the scalars
    it stored, rather than from RoPE math rewritten here. A subtly wrong
    reimplementation would not crash — it would retrieve slightly worse, which
    in a sweep comparing encoders is indistinguishable from a worse encoder.
    """
    embeddings = model[0].auto_model.embeddings
    rotary = embeddings.rotary_emb
    type(rotary).__init__(
        rotary,
        dim=rotary.dim,
        max_position_embeddings=rotary.max_position_embeddings,
        base=rotary.base,
        device=rotary.inv_freq.device,
    )
    position_ids = embeddings.position_ids
    embeddings.register_buffer(
        "position_ids",
        torch.arange(position_ids.size(0), device=position_ids.device),
        persistent=False,
    )

    # Assert rather than trust: every symptom of the original fault was a
    # plausible-looking number somewhere else, so a repair that silently did
    # nothing would reproduce exactly the bug it is here to fix.
    off_unit_circle = float(
        (rotary.cos_cached**2 + rotary.sin_cached**2 - 1).abs().max()
    )
    if off_unit_circle > 1e-4:
        raise RuntimeError(
            "rotary buffers still wrong after repair: cos²+sin² off by "
            f"{off_unit_circle}"
        )


def load_encoder(model_key: str, device: str) -> SentenceTransformer:
    """The encoder for `model_key`, loaded the one way every caller must load it.

    The corpus and the queries searching it have to be encoded by an
    identically configured model. Two construction sites drifting apart would
    not raise anywhere — it would just retrieve badly, and read as the model
    being bad rather than as the two halves disagreeing.
    """
    cfg = MODELS[model_key]
    model = SentenceTransformer(
        cfg["hf_name"],
        device=device,
        trust_remote_code=cfg.get("trust_remote_code", False),
        config_kwargs=cfg.get("config_kwargs") or {},
    )
    if cfg.get("repair_gte_buffers"):
        _repair_gte_buffers(model)
    return model


def chunk_metadata(**fields) -> dict:
    """Vector metadata, minus anything unset.

    Pinecone rejects null metadata values outright, and page_num/year are
    both genuinely optional (a chunk with no provenance, a preprint with no
    year), so an absent key is the only way to express "unknown" - a
    sentinel like 0 or -1 would silently match a `year >= 2023` style filter
    the wrong way round.
    """
    return {k: v for k, v in fields.items() if v is not None}


class Reranker:
    def __init__(self, model_id: str | None = None, device: str = "cpu"):
        from config import load

        self._model = CrossEncoder(
            model_id or load().search.reranker_model, device=device
        )

    def rerank(
        self, query: str, chunks: list[dict], top_k: int | None = None
    ) -> list[dict]:
        pairs = [(query, c["text"]) for c in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        # {**c, "score": ...}, not {"score": ..., **c}: the chunk already has a
        # "score" key (cosine, ts_rank or RRF depending on the mode), so
        # spreading it last silently overwrote the cross-encoder score with the
        # pre-rerank one. Ordering was still correct - it sorts on `scores` -
        # but every reranked result reported the score it had before
        # reranking, which made the number meaningless to threshold on or
        # display.
        results = [{**c, "score": float(s)} for s, c in ranked]
        return results[:top_k] if top_k else results


class PdfEmbedder(PostgresInterface):
    def __init__(self, model_key: str | None = None, namespace: str | None = None):
        from config import load

        super().__init__()
        config = load()
        cfg = MODELS[model_key or config.embedder.model]
        self._cfg = cfg
        # Defaults to the configured namespace so the embed worker and the
        # search engine cannot drift apart - writing to "" while search reads
        # "eval" would produce an index that is silently always empty. Tests
        # still pass "test" explicitly.
        self._namespace = config.search.namespace if namespace is None else namespace
        self._encoder = load_encoder(
            model_key or config.embedder.model, config.devices.embedder
        )
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
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
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

    def pending(self, model_id: int) -> list[tuple[int, str, dict]]:
        """Chunks not yet embedded under this model, each with the metadata
        that goes into the vector alongside it.

        Joins through to documents so the vector carries doc_id and year -
        Pinecone can filter on metadata, and without these the only way to
        answer "search within this paper" or "since 2023" was to over-fetch
        and discard in Postgres afterwards.
        """
        already_embedded = select(ChunkEmbedding.chunk_id).where(
            ChunkEmbedding.model_id == model_id
        )
        stmt = (
            select(
                Chunk.id,
                Chunk.chunk_text,
                Chunk.page_num,
                Document.id.label("doc_id"),
                Document.year,
                Document.corpus,
            )
            .join(Object, Chunk.obj_id == Object.id)
            .join(Document, Object.doc_id == Document.id)
            .where(Chunk.chunk_text.is_not(None))
            .where(Chunk.id.not_in(already_embedded))
        )
        with Session(self.engine) as session:
            rows = session.execute(stmt).all()
        logger.info(f"{len(rows)} chunks pending embedding (model_id={model_id})")
        return [
            (
                r.id,
                r.chunk_text,
                # corpus rides along as metadata even though isolation is by
                # namespace: it makes a mis-filed vector visible in the
                # Pinecone console instead of only in a wrong metric.
                chunk_metadata(
                    page_num=r.page_num, doc_id=r.doc_id, year=r.year, corpus=r.corpus
                ),
            )
            for r in rows
        ]

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._encoder.encode(
            texts,
            batch_size=self._cfg["batch_size"],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def _upsert_vectors(self, batch: list[tuple[int, np.ndarray, dict]]):
        index = self._pc.Index(self._cfg["index_name"])
        vectors = [
            {"id": str(chunk_id), "values": vec.tolist(), "metadata": meta}
            for chunk_id, vec, meta in batch
        ]
        index.upsert(vectors=vectors, namespace=self._namespace)  # type: ignore

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
            metadatas = [r[2] for r in batch]

            vecs = self._embed(texts)
            self._upsert_vectors(list(zip(chunk_ids, vecs, metadatas)))
            self._record_embeddings(chunk_ids, model_id)
            logger.info(f"Embedded {i + len(batch)}/{len(pending)} chunks")

        logger.info(f"Done — {len(pending)} chunks embedded")
