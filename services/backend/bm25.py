"""BM25 over the chunk corpus, as a retrieval mode of its own.

The keyword arm was Postgres `ts_rank` in its two-argument form, which means
`normalization = 0` - no document-length normalisation - and `ts_rank` carries
no IDF at all, so a term appearing in every chunk counts as much as one
appearing in three. Both of BM25's defining components were therefore missing,
and "keyword beats dense" was a claim about an unnamed ranker.

This adds the real thing so the two can be measured against each other rather
than one standing in for the other.

Tokenisation matches Postgres on purpose
----------------------------------------
Postgres' `english` text-search configuration lowercases, drops stopwords and
stems with Snowball. This module does the same three things, so the only thing
that differs between the `keyword` and `bm25` arms is **the ranking function**.
Skipping the stemmer would have handicapped BM25 on morphology - "quantized"
failing to match "quantization" - and the ablation would have measured
tokenisation rather than ranking, which is not the question.

Where the statistics come from
------------------------------
IDF is fitted over **every chunk in the corpus**, not over the candidate pool.
That is what makes it BM25: term rarity is a property of the collection, and
fitting it on a slice would make a term's weight depend on which query pulled
the slice.

Retrieval is two-stage. Postgres FTS finds the chunks that contain any query
lexeme - it has the GIN index and we are not going to beat it in Python - and
BM25 then ranks that pool. The pool has to be wide enough that BM25's ordering
is not capped by `ts_rank`'s; `bm25_pool` defaults to 200 for that reason, and
`eval/ablations.py` sweeps it to confirm the metric has plateaued.

The fitted index is held in memory and cached to disk. That is right for a
corpus of ~11,500 chunks and wrong for one that keeps growing; whether
production switches off `ts_rank` is a question for the measured numbers, not
for this docstring.
"""
import logging
import pickle
import re
from pathlib import Path

import snowballstemmer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import Chunk, Document, Object

logger = logging.getLogger(__name__)

# Classic BM25. rank_bm25 defaults k1 to 1.5; 1.2 is the value from the
# original TREC work and the usual default everywhere else.
K1 = 1.2
B = 0.75

# Letters and digits, so "bge-small" splits but "gpt4" and "8bit" survive
# whole - a papers corpus is full of model names carrying their size.
_TOKEN = re.compile(r"[a-z0-9]+")
_stemmer = snowballstemmer.stemmer("english")


def tokenize(text: str) -> list[str]:
    """Lowercase, split, drop stopwords, stem - the same three steps Postgres'
    `english` configuration applies, so the two keyword arms differ only in
    how they score."""
    words = [w for w in _TOKEN.findall(text.lower()) if w not in ENGLISH_STOP_WORDS]
    return _stemmer.stemWords(words)


def corpus_rows(session, corpus: str | None) -> list[tuple[int, str]]:
    """Every chunk the corpus contains, in id order.

    Ordered so the index is reproducible: BM25Okapi scores by position, and a
    cached index whose positions no longer line up with its chunk ids would
    return confident, wrong chunks rather than failing.
    """
    stmt = (
        select(Chunk.id, Chunk.chunk_text)
        .join(Object, Chunk.obj_id == Object.id)
        .join(Document, Object.doc_id == Document.id)
        .where(Chunk.chunk_text.is_not(None))
        .order_by(Chunk.id)
    )
    if corpus is not None:
        stmt = stmt.where(Document.corpus == corpus)
    return [(r.id, r.chunk_text) for r in session.execute(stmt).all()]


def fingerprint(session, corpus: str | None) -> str:
    """Cheap identity for the corpus as it stands now.

    Count plus max id: re-chunking renumbers from a new high-water mark, so
    either number moving means the cache is stale. Hashing 11,500 chunk texts
    on every search would cost more than fitting the index.
    """
    stmt = (
        select(func.count(Chunk.id), func.max(Chunk.id))
        .join(Object, Chunk.obj_id == Object.id)
        .join(Document, Object.doc_id == Document.id)
    )
    if corpus is not None:
        stmt = stmt.where(Document.corpus == corpus)
    count, high = session.execute(stmt).one()
    return f"{count}-{high}"


class Bm25Index:
    """A fitted BM25 over one corpus, plus the chunk ids its positions mean."""

    def __init__(self, chunk_ids: list[int], model: BM25Okapi, fingerprint: str):
        self.chunk_ids = chunk_ids
        self.model = model
        self.fingerprint = fingerprint
        self._position = {cid: i for i, cid in enumerate(chunk_ids)}

    def __len__(self) -> int:
        return len(self.chunk_ids)

    @classmethod
    def fit(cls, session, corpus: str | None) -> "Bm25Index":
        rows = corpus_rows(session, corpus)
        if not rows:
            raise ValueError(f"no chunks to fit BM25 on for corpus {corpus!r}")
        ids = [cid for cid, _ in rows]
        model = BM25Okapi([tokenize(text) for _, text in rows], k1=K1, b=B)
        logger.info(f"fitted BM25 over {len(ids)} chunks (corpus={corpus!r})")
        return cls(ids, model, fingerprint(session, corpus))

    def scores_for(self, query: str, chunk_ids: list[int]) -> dict[int, float]:
        """BM25 score for each of these chunks, IDF from the whole collection.

        `get_batch_scores` rather than `get_scores`: scoring the candidate pool
        is a couple of hundred dot products, scoring the collection is 11,500
        for a result that is then thrown away.

        Chunks the index has never seen are skipped rather than scored zero - a
        zero is a real BM25 score meaning "no query term present", and a chunk
        that postdates the cache has no score at all, which is a different
        thing and should not silently rank above one that genuinely matched
        nothing.
        """
        known = [(cid, self._position[cid]) for cid in chunk_ids if cid in self._position]
        if not known:
            return {}
        tokens = tokenize(query)
        if not tokens:
            return {}
        scores = self.model.get_batch_scores(tokens, [pos for _, pos in known])
        return {cid: float(s) for (cid, _), s in zip(known, scores)}


def cache_path(root: str | Path, corpus: str | None) -> Path:
    return Path(root) / "bm25" / f"{corpus or 'all'}.pkl"


def load_or_fit(session, corpus: str | None, root: str | Path) -> Bm25Index:
    """The cached index if it still describes this corpus, otherwise a fresh
    fit written back to disk.

    Fitting is seconds over ~11,500 chunks, so this is about not paying it on
    every process start rather than about it being slow.
    """
    path = cache_path(root, corpus)
    current = fingerprint(session, corpus)
    if path.exists():
        try:
            index = pickle.loads(path.read_bytes())
            if index.fingerprint == current:
                logger.info(f"loaded BM25 index from {path} ({len(index)} chunks)")
                return index
            logger.info(f"BM25 cache stale ({index.fingerprint} != {current}), refitting")
        except Exception as e:
            # A cache that cannot be read is a cache miss, never a failed
            # search - the index is derived data and refitting is cheap.
            logger.warning(f"BM25 cache at {path} unreadable ({e}), refitting")

    index = Bm25Index.fit(session, corpus)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(index))
    return index
