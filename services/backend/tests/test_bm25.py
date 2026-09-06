"""BM25 as a retrieval mode - pure functions and a fitted index, no infra.

The point of this arm is that `keyword` was never BM25: ts_rank has no IDF
and, in its two-argument form, no length normalisation. These tests pin down
the two properties that difference is made of, because a BM25 that quietly
lost either would look like a working comparison and measure nothing.
"""
import bm25s
import pytest

from bm25 import METHOD, Bm25Index, cache_path, tokenize


def terms(text):
    return tokenize(text, is_query=True)[0]


class TestTokenize:
    """bm25s' standard English analyzer - lowercase, stopwords, Snowball -
    which is the same set of steps Postgres' `english` configuration applies.
    Matching it is what leaves the ranking function as the only difference
    between the keyword and bm25 arms."""

    def test_lowercases_and_splits(self):
        assert terms("Dense Retrieval") == terms("dense retrieval")

    def test_stems_so_morphology_is_not_the_thing_being_measured(self):
        """Postgres stems; if this did not, BM25 would lose "quantized"
        against "quantization" and the ablation would be measuring
        tokenisation rather than ranking."""
        assert terms("quantization") == terms("quantized")

    def test_drops_stopwords(self):
        assert "the" not in terms("the model")

    def test_keeps_digits_attached_to_their_word(self):
        """A papers corpus is full of model names carrying their size, and
        splitting gpt4 into gpt and 4 loses the name."""
        assert "gpt4" in terms("GPT4 results")

    def test_punctuation_never_becomes_a_token(self):
        assert terms("bge-small, v2.") == terms("bge small v2")


# BM25's IDF is a property of the whole collection and degenerates on tiny
# ones - with two documents, a term in one of them carries almost no
# information by definition. That is real BM25 behaviour rather than a bug,
# but it makes a three-document fixture measure the degenerate case, so these
# tests pad to a realistic collection where every document is distinct.
BACKGROUND = [f"unrelated filler document number {i}" for i in range(30)]


def _index(texts, start_id=1):
    """A fitted index over `texts` plus background documents.

    The background is what gives the query terms a sane document frequency;
    without it every idf is computed over a handful of documents and the
    scores say nothing about ranking.
    """
    corpus = list(texts) + BACKGROUND
    ids = list(range(start_id, start_id + len(corpus)))
    retriever = bm25s.BM25(method=METHOD)
    retriever.index(tokenize(corpus), show_progress=False)
    return Bm25Index(ids, retriever, fingerprint="test")


class TestBm25Index:
    def test_scores_the_chunk_containing_the_query_term_highest(self):
        index = _index(["retrieval augmented generation", "protein folding assay"])

        scores = index.scores_for("retrieval augmented", [1, 2])

        assert scores[1] > scores[2]

    def test_rare_terms_outweigh_common_ones(self):
        """This is the IDF ts_rank does not have at all. "model" is in half
        the collection and carries almost no information; "quantization" is in
        one chunk. A ranker without IDF cannot tell these two apart."""
        common = ["model model model model"] * 20
        index = _index(["model model quantization"] + common)

        scores = index.scores_for("model quantization", [1, 2])

        assert scores[1] > scores[2]

    def test_a_longer_chunk_is_not_rewarded_for_being_longer(self):
        """The length normalisation ts_rank's two-argument form omits. Both
        chunks contain the term once; the padded one must not win."""
        index = _index(["sparse attention", "sparse attention " + "filler " * 60])

        scores = index.scores_for("sparse attention", [1, 2])

        assert scores[1] > scores[2]

    def test_only_the_requested_chunks_are_scored(self):
        """Retrieval rescores an FTS pool of a couple of hundred, not the
        whole collection - get_batch_scores, not get_scores."""
        index = _index(["alpha", "beta", "gamma"])

        assert set(index.scores_for("alpha beta", [1, 3])) == {1, 3}

    def test_chunks_the_index_has_never_seen_are_skipped_not_zeroed(self):
        """Zero is a real BM25 score meaning "no query term present". A chunk
        that postdates the cache has no score at all, and conflating the two
        would rank an unknown chunk above one that genuinely matched
        nothing."""
        index = _index(["alpha"])

        assert index.scores_for("alpha", [1, 999]) == {1: pytest.approx(index.scores_for("alpha", [1])[1])}

    def test_a_query_of_pure_stopwords_scores_nothing(self):
        """Tokenising to an empty list must not be handed to the ranker."""
        index = _index(["alpha"])

        assert index.scores_for("the of and", [1]) == {}

    def test_fitting_an_empty_corpus_fails_loudly(self):
        """An index over nothing scores everything zero, which reads as
        "retrieval found nothing relevant" on every question forever."""
        with pytest.raises(ValueError, match="no chunks"):
            Bm25Index.fit(_FakeSession([]), "eval")


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *_):
        class _R:
            def __init__(self, rows):
                self._rows = rows

            def all(self):
                return self._rows

            def one(self):
                return (len(self._rows), max((r.id for r in self._rows), default=0))

        return _R(self._rows)


class TestCachePath:
    def test_keyed_on_the_corpus(self):
        assert cache_path("/data", "eval") != cache_path("/data", "main")

    def test_an_unscoped_corpus_still_has_a_name(self):
        assert cache_path("/data", None).name == "all"


class TestRoundTrip:
    def test_a_saved_index_scores_identically_when_reloaded(self, tmp_path):
        """The cache exists so a process start does not refit. It is only
        useful if the reloaded index ranks the same - a chunk_ids list that
        drifted from the matrix would return confident, wrong chunks."""
        index = _index(["retrieval augmented generation", "protein folding"])
        index.save(tmp_path / "idx")

        reloaded = Bm25Index.load(tmp_path / "idx")

        assert reloaded.scores_for("retrieval", [1, 2]) == index.scores_for(
            "retrieval", [1, 2]
        )
