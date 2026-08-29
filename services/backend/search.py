"""Retrieval over the indexed paper chunks, in three selectable modes.

    semantic  - dense vector search (Pinecone), the original behaviour
    keyword   - Postgres full-text search (tsvector/GIN)
    hybrid    - both of the above, fused with Reciprocal Rank Fusion

The two single-source modes are unchanged; hybrid is additive. Reranking
(cross-encoder) is orthogonal and applies to whichever mode produced the
candidates.
"""
import os

from db.connection import PostgresInterface
from db.models import Chunk, Document, Object
from pinecone.grpc import PineconeGRPC as Pinecone
from process.embedder import MODELS, Reranker
from schemas import ChunkResult, SearchResponse
from sentence_transformers import SentenceTransformer
from sqlalchemy import func, select
from sqlalchemy.orm import Session

SEMANTIC = "semantic"
KEYWORD = "keyword"
HYBRID = "hybrid"
RETRIEVAL_MODES = (SEMANTIC, KEYWORD, HYBRID)


def rrf_fuse(ranked_lists: list[list[dict]], k: int = 60, key: str = "chunk_id") -> list[dict]:
    """Reciprocal Rank Fusion: score each chunk as sum(1 / (k + rank)) over
    the lists it appears in, ranks 1-based.

    Fuses *ranks*, not scores, on purpose. A cosine similarity from Pinecone
    and a Postgres `ts_rank` are on incomparable scales - blending them
    directly needs normalisation constants that are themselves a tuning
    problem, and those constants shift whenever the corpus or the embedding
    model changes. Ranks are scale-free, so this stays stable across both.

    k dampens how much the very top ranks dominate; 60 is the value from the
    original RRF paper and the usual default.

    Each returned chunk's `score` is replaced by its RRF score - the source
    scores aren't comparable to it, and keeping both invites reading the
    wrong one.
    """
    scores: dict[int, float] = {}
    chunks: dict[int, dict] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            cid = chunk[key]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            # Keep the first-seen copy: the rows are identical either way, and
            # this avoids depending on which source happened to come last.
            chunks.setdefault(cid, chunk)

    return [
        {**chunks[cid], "score": score}
        for cid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]


def _chunk_rows_stmt(chunk_ids=None):
    """The join every retrieval path needs: chunk text + page, the PDF it came
    from, and its parent document's metadata."""
    stmt = (
        select(
            Chunk.id,
            Chunk.chunk_text,
            Chunk.page_num,
            Object.path,
            Document.id.label("doc_id"),
            Document.title,
            Document.authors,
            Document.year,
        )
        .join(Object, Chunk.obj_id == Object.id)
        .join(Document, Object.doc_id == Document.id)
    )
    return stmt if chunk_ids is None else stmt.where(Chunk.id.in_(chunk_ids))


def _row_to_chunk(r, score: float) -> dict:
    return {
        "chunk_id": r.id,
        "doc_id": r.doc_id,
        "text": r.chunk_text,
        "page_num": r.page_num,
        "pdf_path": r.path,
        "title": r.title,
        "authors": r.authors,
        "year": r.year,
        "score": score,
    }


def _keyword_rows(session, query: str, top_k: int) -> list[dict]:
    """Postgres full-text search over chunk_text (tsvector/GIN, not a separate
    search service). Shared by the standalone keyword_search() below and by
    the engine's keyword/hybrid modes, so all of them rank identically."""
    tsquery = func.plainto_tsquery("english", query)
    tsvector = func.to_tsvector("english", Chunk.chunk_text)
    rank = func.ts_rank(tsvector, tsquery)

    stmt = (
        _chunk_rows_stmt()
        .add_columns(rank.label("score"))
        .where(tsvector.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(top_k)
    )
    return [_row_to_chunk(r, float(r.score)) for r in session.execute(stmt).all()]


class SearchEngine(PostgresInterface):
    def __init__(
        self,
        encoder: SentenceTransformer,
        reranker: Reranker,
        model_key: str = "bge-small",
    ):
        super().__init__()
        self._cfg = MODELS[model_key]
        self._model_key = model_key
        self._encoder = encoder
        self._reranker = reranker
        self._pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    def _embed_query(self, query: str) -> list[float]:
        return self._encoder.encode(
            query,
            prompt=self._cfg["query_prompt"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

    def _vector_candidates(self, query: str, top_k: int) -> list[dict]:
        vec = self._embed_query(query)
        index = self._pc.Index(self._cfg["index_name"])
        matches = index.query(vector=vec, top_k=top_k, include_metadata=True)["matches"]
        if not matches:
            return []

        scores = {int(m["id"]): m["score"] for m in matches}
        with Session(self.engine) as session:
            rows = session.execute(_chunk_rows_stmt(list(scores))).all()

        chunks = [_row_to_chunk(r, scores[r.id]) for r in rows]
        # Pinecone returns ranked results; the SQL hydration doesn't preserve
        # that order, so restore it before anything downstream reads ranks.
        chunks.sort(key=lambda c: c["score"], reverse=True)
        return chunks

    def _keyword_candidates(self, query: str, top_k: int) -> list[dict]:
        with Session(self.engine) as session:
            return _keyword_rows(session, query, top_k)

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = False,
        rerank_top_k: int = 5,
        mode: str | None = None,
    ) -> SearchResponse:
        from config import load

        cfg = load().search
        mode = mode or cfg.mode
        if mode not in RETRIEVAL_MODES:
            raise ValueError(
                f"unknown retrieval mode {mode!r} (expected one of {RETRIEVAL_MODES})"
            )

        if mode == SEMANTIC:
            chunks = self._vector_candidates(query, top_k)
        elif mode == KEYWORD:
            chunks = self._keyword_candidates(query, top_k)
        else:
            # Each source contributes a wider pool than the final top_k -
            # pulling only top_k per source would let one source's misses cap
            # what fusion has to work with.
            pool = max(cfg.hybrid_candidates, top_k)
            chunks = rrf_fuse(
                [self._vector_candidates(query, pool), self._keyword_candidates(query, pool)],
                k=cfg.rrf_k,
            )[:top_k]

        if rerank and chunks:
            chunks = self._reranker.rerank(query, chunks, top_k=rerank_top_k)

        return SearchResponse(
            query=query,
            model=self._model_key,
            mode=mode,
            reranked=rerank and bool(chunks),
            results=[ChunkResult(**c) for c in chunks],
        )


def keyword_search(query: str, top_k: int = 10) -> SearchResponse:
    """Standalone keyword search, backing the /search/keyword endpoint.

    Kept as a module function rather than folded into SearchEngine because it
    needs neither the encoder nor the reranker - both expensive to load - so
    the endpoint stays usable from a process that never loads them.
    """
    with Session(PostgresInterface.connect()) as session:
        results = _keyword_rows(session, query, top_k)
    return SearchResponse(
        query=query,
        model="keyword",
        mode=KEYWORD,
        reranked=False,
        results=[ChunkResult(**c) for c in results],
    )


_engine: SearchEngine | None = None


def get_search_engine() -> SearchEngine:
    """Lazy singleton, shared between the /search endpoint and the
    orchestrator's RAG tools - the encoder + reranker are expensive to load,
    loading them twice (once per caller) would be wasteful."""
    global _engine
    if _engine is None:
        from config import load

        cfg = load()
        model_key = cfg.embedder.model
        encoder = SentenceTransformer(
            MODELS[model_key]["hf_name"], device=cfg.devices.embedder
        )
        reranker = Reranker(device=cfg.devices.reranker)
        _engine = SearchEngine(encoder=encoder, reranker=reranker, model_key=model_key)
    return _engine
