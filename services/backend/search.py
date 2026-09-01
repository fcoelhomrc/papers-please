"""Retrieval over the indexed paper chunks, in three selectable modes.

    semantic  - dense vector search (Pinecone), the original behaviour
    keyword   - Postgres full-text search (tsvector/GIN)
    hybrid    - both of the above, fused with Reciprocal Rank Fusion

The two single-source modes are unchanged; hybrid is additive. Reranking
(cross-encoder) is orthogonal and applies to whichever mode produced the
candidates.
"""
import os
from pathlib import Path

from db.connection import PostgresInterface
from db.models import Chunk, Document, Object
from pinecone.grpc import PineconeGRPC as Pinecone
from process.embedder import MODELS, Reranker
from schemas import ChunkResult, SearchResponse
from sentence_transformers import SentenceTransformer
from sqlalchemy import Text, func, literal_column, select, tuple_
from sqlalchemy.orm import Session

SEMANTIC = "semantic"
KEYWORD = "keyword"
HYBRID = "hybrid"
RETRIEVAL_MODES = (SEMANTIC, KEYWORD, HYBRID)


def rrf_fuse(
    ranked_lists: list[list[dict]],
    k: int = 60,
    key: str = "chunk_id",
    weights: list[float] | None = None,
    labels: list[str] | None = None,
) -> list[dict]:
    """Reciprocal Rank Fusion: score each chunk as sum(w / (k + rank)) over
    the lists it appears in, ranks 1-based.

    Fuses *ranks*, not scores, on purpose. A cosine similarity from Pinecone
    and a Postgres `ts_rank` are on incomparable scales - blending them
    directly needs normalisation constants that are themselves a tuning
    problem, and those constants shift whenever the corpus or the embedding
    model changes. Ranks are scale-free, so this stays stable across both.

    k dampens how much the very top ranks dominate; 60 is the value from the
    original RRF paper and the usual default.

    `weights` scales each list's contribution. Plain RRF weights every source
    equally, which assumes they are comparably good rankers - and ours are
    not: measured alone at top_k=5, dense retrieval scores nDCG 0.787 while
    keyword scores 0.598, so weighting them equally drags the fused ranking
    below dense-only (0.756). Down-weighting keyword makes it a tiebreaker
    that adds its hits without outvoting the stronger ranker.

    Each returned chunk's `score` is replaced by its RRF score - the source
    scores aren't comparable to it, and keeping both invites reading the
    wrong one.

    `labels` names each list, and each fused chunk records which of them found
    it in `sources`. Fusion is the only place that knows this - afterwards a
    chunk found by both retrievers is indistinguishable from one found by
    either, and "why did this match?" becomes unanswerable. It's also the
    most useful thing to show next to a hybrid result: agreement between two
    independent retrievers is a different kind of confidence from one
    retriever being very sure.
    """
    weights = weights or [1.0] * len(ranked_lists)
    labels = labels or [str(i) for i in range(len(ranked_lists))]
    scores: dict[int, float] = {}
    chunks: dict[int, dict] = {}
    sources: dict[int, list[str]] = {}
    for ranked, weight, label in zip(ranked_lists, weights, labels):
        for rank, chunk in enumerate(ranked, start=1):
            cid = chunk[key]
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
            # Keep the first-seen copy: the rows are identical either way, and
            # this avoids depending on which source happened to come last.
            chunks.setdefault(cid, chunk)
            sources.setdefault(cid, []).append(label)

    return [
        {**chunks[cid], "score": score, "sources": sources[cid]}
        for cid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]


def _chunk_rows_stmt(chunk_ids=None):
    """The join every retrieval path needs: chunk text + page, the PDF it came
    from, and its parent document's metadata."""
    stmt = (
        select(
            Chunk.id,
            Chunk.chunk_index,
            Chunk.obj_id,
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


def _pdf_exists(path: str | None) -> bool:
    """Whether the chunk's source PDF is actually readable.

    The result card uses this to decide whether to offer a Preview. A chunk
    can exist with no file behind it - eval fixtures are seeded as chunk
    text against a synthetic path - and offering a preview then produces a
    button that can only 404.
    """
    if not path:
        return False
    from config import load

    try:
        return (Path(load().storage.root) / path).is_file()
    except OSError:
        # An unreadable or misconfigured storage root is a "no", not a 500
        # on a search request.
        return False


def _row_to_chunk(r, score: float) -> dict:
    return {
        "chunk_id": r.id,
        "chunk_index": r.chunk_index,
        # Not part of ChunkResult - only neighbour expansion needs it, and it
        # is dropped before the response is built.
        "obj_id": r.obj_id,
        "doc_id": r.doc_id,
        "text": r.chunk_text,
        "page_num": r.page_num,
        "pdf_path": r.path,
        "has_pdf": _pdf_exists(r.path),
        "title": r.title,
        "authors": r.authors,
        "year": r.year,
        "score": score,
    }


def _or_tsquery(query: str):
    """A tsquery matching ANY of the query's lexemes, not all of them.

    plainto_tsquery ANDs every term, so a natural-language question only
    matches a chunk containing all of its words - which measured out as
    "returns anything for 9 of 50 eval questions, recall flat at 0.167 from
    k=3": a matching failure, not a ranking one.

    Rewriting the operators to | keeps everything plainto_tsquery does well
    (stemming, stopword removal, escaping - the input never reaches tsquery
    syntax unsanitised) and only relaxes the predicate. ts_rank already
    scores a chunk matching more terms above one matching fewer, so ranking
    does the discrimination the match predicate was over-doing.
    """
    plain = func.plainto_tsquery("english", query)
    return func.replace(func.cast(plain, Text), "&", "|").op("::")(literal_column("tsquery"))


def _keyword_rows(session, query: str, top_k: int, min_score: float | None = None) -> list[dict]:
    """Postgres full-text search over chunk_text (tsvector/GIN, not a separate
    search service). Shared by the standalone keyword_search() below and by
    the engine's keyword/hybrid modes, so all of them rank identically."""
    tsquery = _or_tsquery(query)
    tsvector = func.to_tsvector("english", Chunk.chunk_text)
    rank = func.ts_rank(tsvector, tsquery)

    stmt = (
        _chunk_rows_stmt()
        .add_columns(rank.label("score"))
        .where(tsvector.op("@@")(tsquery))
    )
    if min_score is not None:
        stmt = stmt.where(rank >= min_score)
    stmt = stmt.order_by(rank.desc()).limit(top_k)
    return [_row_to_chunk(r, float(r.score)) for r in session.execute(stmt).all()]


def expand_neighbours(session, chunks: list[dict], window: int) -> list[dict]:
    """Glue each hit's neighbouring chunks onto it as `context`.

    Small-to-big: the vector is built from one tight chunk so it means one
    thing, and the prose handed to a reader is the window around it. This runs
    *after* ranking on purpose - expanding first would mean the cross-encoder
    scored a blur of three chunks and the score would no longer say which one
    matched.

    One batched query for every neighbour of every hit, keyed on
    (obj_id, chunk_index): a query per hit is 5-10 round trips per search on
    the request path, and this is the same cost regardless of how many hits
    came back.

    Neighbours are looked up within the same `obj_id`, not the same document:
    chunk_index restarts per PDF, so a document with two objects would
    otherwise splice one paper's text into another's.
    """
    if window <= 0 or not chunks:
        return chunks

    wanted = {
        (c["obj_id"], c["chunk_index"] + offset)
        for c in chunks
        for offset in range(-window, window + 1)
        if c["chunk_index"] + offset >= 0
    }
    rows = session.execute(
        select(Chunk.obj_id, Chunk.chunk_index, Chunk.chunk_text).where(
            tuple_(Chunk.obj_id, Chunk.chunk_index).in_(wanted)
        )
    ).all()
    by_position = {(r.obj_id, r.chunk_index): r.chunk_text for r in rows}

    expanded = []
    for c in chunks:
        parts = [
            by_position.get((c["obj_id"], c["chunk_index"] + offset))
            for offset in range(-window, window + 1)
        ]
        # Missing neighbours are normal at a document's edges, and gaps close
        # up rather than leaving blank joins in the middle of the prose.
        text = "\n\n".join(p for p in parts if p)
        expanded.append({**c, "context": text or c["text"]})
    return expanded


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

    def _vector_candidates(self, query: str, top_k: int, min_score: float | None = None) -> list[dict]:
        vec = self._embed_query(query)
        index = self._pc.Index(self._cfg["index_name"])
        matches = index.query(vector=vec, top_k=top_k, include_metadata=True)["matches"]
        if not matches:
            return []

        scores = {int(m["id"]): m["score"] for m in matches}
        if min_score is not None:
            scores = {cid: sc for cid, sc in scores.items() if sc >= min_score}
            if not scores:
                return []
        with Session(self.engine) as session:
            rows = session.execute(_chunk_rows_stmt(list(scores))).all()

        chunks = [_row_to_chunk(r, scores[r.id]) for r in rows]
        # Pinecone returns ranked results; the SQL hydration doesn't preserve
        # that order, so restore it before anything downstream reads ranks.
        chunks.sort(key=lambda c: c["score"], reverse=True)
        return chunks

    def _keyword_candidates(self, query: str, top_k: int, min_score: float | None = None) -> list[dict]:
        with Session(self.engine) as session:
            return _keyword_rows(session, query, top_k, min_score=min_score)

    def _expand(self, chunks: list[dict], window: int) -> list[dict]:
        with Session(self.engine) as session:
            return expand_neighbours(session, chunks, window)

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = False,
        rerank_top_k: int = 5,
        mode: str | None = None,
        thresholds: dict | None = None,
        candidates: int | None = None,
        neighbour_window: int | None = None,
    ) -> SearchResponse:
        """`candidates` widens retrieval ahead of the reranker: fetch this
        many, then let the cross-encoder cut to `rerank_top_k`. Without it,
        retrieval fetches exactly `top_k` and reranking can only reorder what
        it was given - never promote a chunk that ranked 12th into the top 5,
        which is most of what a cross-encoder is for.

        `neighbour_window` widens each surviving hit with the chunks either
        side of it, exposed as `context` while `text` stays the chunk that
        actually matched. Also opt-in, and also applied only at the end.

        Opt-in rather than the default because `top_k` means "how many to
        retrieve" to callers that are measuring retrieval itself (eval.sweep,
        the /search endpoint's explicit knobs); silently retrieving 40 when
        they asked for 5 would corrupt what they are measuring. Callers whose
        intent is "give me the best k" - search_chunks, FixedPipeline - pass
        it. Ignored when `rerank` is off, since there would then be nothing
        to narrow the pool back down.

        `thresholds` overrides the configured minimum scores, keyed
        min_vector_score / min_keyword_score / min_rerank_score. They are in
        each source's own units on purpose - cosine (~0.5-0.9), ts_rank
        (~0.0-0.1), cross-encoder logits (~-11..+11) and RRF (~0.016) are not
        on comparable scales, so one shared number would mean four different
        things. Passing a threshold lets retrieval return nothing, which is
        the correct answer when the library has nothing relevant."""
        from config import load

        cfg = load().search
        mode = mode or cfg.mode
        if mode not in RETRIEVAL_MODES:
            raise ValueError(
                f"unknown retrieval mode {mode!r} (expected one of {RETRIEVAL_MODES})"
            )

        t = {
            "min_vector_score": cfg.min_vector_score,
            "min_keyword_score": cfg.min_keyword_score,
            "min_rerank_score": cfg.min_rerank_score,
            **(thresholds or {}),
        }

        # What retrieval fetches, which is only the same as top_k when the
        # reranker isn't there to narrow it afterwards.
        retrieve_k = max(candidates, top_k) if (rerank and candidates) else top_k

        if mode == SEMANTIC:
            chunks = [
                {**c, "sources": [SEMANTIC]}
                for c in self._vector_candidates(query, retrieve_k, t["min_vector_score"])
            ]
        elif mode == KEYWORD:
            chunks = [
                {**c, "sources": [KEYWORD]}
                for c in self._keyword_candidates(query, retrieve_k, t["min_keyword_score"])
            ]
        else:
            # Each source contributes a wider pool than the final top_k -
            # pulling only top_k per source would let one source's misses cap
            # what fusion has to work with.
            pool = max(cfg.hybrid_candidates, retrieve_k)
            # Thresholds apply per source, before fusion: an RRF score is a
            # rank artefact with no notion of "relevant enough", so filtering
            # after fusion could not express this at all.
            chunks = rrf_fuse(
                [
                    self._vector_candidates(query, pool, t["min_vector_score"]),
                    self._keyword_candidates(query, pool, t["min_keyword_score"]),
                ],
                k=cfg.rrf_k,
                weights=[1.0, cfg.keyword_weight],
                labels=[SEMANTIC, KEYWORD],
            )[:retrieve_k]

        # Tracked separately from `chunks` being non-empty: with a rerank
        # floor, an empty result can now mean "reranked, and nothing cleared
        # the bar" as well as "nothing was retrieved". Deriving the flag from
        # emptiness would report reranked=False for the former, which is the
        # opposite of what happened.
        did_rerank = bool(rerank and chunks)
        if did_rerank:
            chunks = self._reranker.rerank(query, chunks, top_k=rerank_top_k)
            if t["min_rerank_score"] is not None:
                chunks = [c for c in chunks if c["score"] >= t["min_rerank_score"]]

        # Last, on the handful of chunks that survived - expanding the
        # candidate pool instead would fetch neighbours for 40 chunks to
        # throw 35 of them away.
        window = cfg.neighbour_window if neighbour_window is None else neighbour_window
        if window and chunks:
            chunks = self._expand(chunks, window)

        return SearchResponse(
            query=query,
            model=self._model_key,
            mode=mode,
            reranked=did_rerank,
            # obj_id is an internal join key that neighbour expansion needs
            # and the API contract doesn't have.
            results=[ChunkResult(**{k: v for k, v in c.items() if k != "obj_id"}) for c in chunks],
        )


def keyword_search(query: str, top_k: int = 10) -> SearchResponse:
    """Standalone keyword search, backing the /search/keyword endpoint.

    Kept as a module function rather than folded into SearchEngine because it
    needs neither the encoder nor the reranker - both expensive to load - so
    the endpoint stays usable from a process that never loads them.
    """
    with Session(PostgresInterface.connect()) as session:
        results = [{**c, "sources": [KEYWORD]} for c in _keyword_rows(session, query, top_k)]
    return SearchResponse(
        query=query,
        model="keyword",
        mode=KEYWORD,
        reranked=False,
        results=[
            ChunkResult(**{k: v for k, v in c.items() if k != "obj_id"}) for c in results
        ],
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
