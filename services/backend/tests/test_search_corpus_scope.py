"""Corpus isolation: what retrieval is allowed to see.

The eval corpus has to be separable from anything fetched ad-hoc through the
UI. One stray paper in the haystack changes what every question is competing
against, and every number measured before it silently stops being comparable.

Isolation is enforced in two places because the two retrievers fail
differently - SQL filtering for keyword, Pinecone namespace for dense. These
tests pin both, and pin the reason the dense side cannot use a post-filter.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import config as config_module
import pytest
from config import Config, SearchConfig
from search import SearchEngine, _chunk_rows_stmt, _corpus_filter


@pytest.fixture
def scoped(monkeypatch):
    """Config pinned to an 'eval' corpus, restored afterwards."""
    monkeypatch.setattr(
        config_module,
        "_config",
        Config(search=SearchConfig(corpus="eval", namespace="eval")),
    )


class TestCorpusFilter:
    def test_unset_resolves_from_config(self, scoped):
        """Callers that don't name a corpus get the configured one, so a new
        retrieval path cannot accidentally opt out of isolation."""
        from search import _UNSET

        assert _corpus_filter(_UNSET) == "eval"

    def test_explicit_none_means_every_corpus(self, scoped):
        """None is a real value here - 'search everything' - so it must not
        fall through to the config default."""
        assert _corpus_filter(None) is None

    def test_explicit_value_overrides_config(self, scoped):
        assert _corpus_filter("main") == "main"


class TestChunkRowsStmt:
    def test_scopes_the_join_to_the_configured_corpus(self, scoped):
        """The filter lives in the shared statement builder, not in search(),
        because the ablation harness calls the candidate methods directly."""
        assert "documents.corpus" in str(_chunk_rows_stmt())

    def test_no_filter_when_corpus_is_none(self):
        assert "documents.corpus" not in str(_chunk_rows_stmt(corpus=None))


class TestVectorCandidatesIsolation:
    def _engine(self, matches):
        engine = SearchEngine.__new__(SearchEngine)
        engine._cfg = {"index_name": "idx", "query_prompt": ""}
        engine._namespace = "eval"
        engine._corpus = "eval"
        engine._embed_query = MagicMock(return_value=[0.0])
        index = MagicMock()
        index.query.return_value = {"matches": matches}
        engine._pc = MagicMock()
        engine._pc.Index.return_value = index
        engine.engine = MagicMock()
        return engine, index

    def test_queries_pinecone_within_the_namespace(self):
        """Isolation for the dense path is the namespace. Querying the default
        namespace and filtering afterwards returns fewer rows than top_k asked
        for, which reads as a recall drop that retrieval never caused."""
        engine, index = self._engine([])

        engine._vector_candidates("q", top_k=5)

        assert index.query.call_args.kwargs["namespace"] == "eval"

    def test_hydration_carries_the_corpus_scope(self):
        engine, _ = self._engine([{"id": "1", "score": 0.9}])
        session = MagicMock()
        session.execute.return_value.all.return_value = []

        with patch("search.Session") as sess:
            sess.return_value.__enter__.return_value = session
            with patch("search._chunk_rows_stmt") as stmt:
                engine._vector_candidates("q", top_k=5)

        assert stmt.call_args.kwargs["corpus"] == "eval"


class TestKeywordCandidatesIsolation:
    def test_passes_the_corpus_scope_through(self):
        engine = SearchEngine.__new__(SearchEngine)
        engine._corpus = "eval"
        engine.engine = MagicMock()

        with patch("search.Session"), patch("search._keyword_rows") as rows:
            rows.return_value = []
            engine._keyword_candidates("q", top_k=5)

        assert rows.call_args.kwargs["corpus"] == "eval"


class TestEmbedderNamespace:
    def test_defaults_to_the_configured_namespace(self, scoped, monkeypatch):
        """The embed worker and the search engine must agree. Writing to ""
        while search reads "eval" produces an index that is silently always
        empty - no error, just zero results forever."""
        from process.embedder import PdfEmbedder

        monkeypatch.setenv("PINECONE_API_KEY", "k")

        with patch("process.embedder.SentenceTransformer"), patch(
            "process.embedder.Pinecone"
        ), patch.object(PdfEmbedder, "connect", return_value=MagicMock()):
            embedder = PdfEmbedder()

        assert embedder._namespace == "eval"

    def test_explicit_namespace_still_wins(self, scoped, monkeypatch):
        """Integration tests pin themselves to a 'test' namespace."""
        from process.embedder import PdfEmbedder

        monkeypatch.setenv("PINECONE_API_KEY", "k")

        with patch("process.embedder.SentenceTransformer"), patch(
            "process.embedder.Pinecone"
        ), patch.object(PdfEmbedder, "connect", return_value=MagicMock()):
            embedder = PdfEmbedder(namespace="test")

        assert embedder._namespace == "test"


class TestVectorMetadataCarriesCorpus:
    def test_pending_tags_each_vector_with_its_corpus(self):
        """Redundant with the namespace, deliberately: a mis-filed vector is
        then visible in the Pinecone console rather than only as a wrong
        number in a metric."""
        from process.embedder import PdfEmbedder

        embedder = PdfEmbedder.__new__(PdfEmbedder)
        row = SimpleNamespace(
            id=1, chunk_text="t", page_num=2, doc_id=3, year=2024, corpus="eval"
        )
        session = MagicMock()
        session.execute.return_value.all.return_value = [row]

        with patch("process.embedder.Session") as sess:
            sess.return_value.__enter__.return_value = session
            embedder.engine = MagicMock()
            pending = embedder.pending(model_id=1)

        assert pending[0][2]["corpus"] == "eval"
