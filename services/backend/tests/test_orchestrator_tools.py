"""Unit tests for orchestrator/tools.py.

Each @tool-decorated function becomes a langchain StructuredTool. We call it
the same way the future agent will - via .invoke({...}) - rather than as a
plain function, so these tests also cover the argument schema langchain
derives from the type hints. Underlying executor classes and DB session are
mocked, so no real Postgres/Pinecone/network involved.
"""
from unittest.mock import MagicMock, patch

from orchestrator import tools


class TestFetchPapers:
    def test_invoke_calls_fetcher_and_reports_count(self):
        with patch("orchestrator.tools.SemanticScholarFetcher") as MockFetcher:
            MockFetcher.return_value.fetch.return_value = 7
            result = tools.fetch_papers.invoke({"query": "transformers", "max_papers": 50})

            MockFetcher.return_value.fetch.assert_called_once_with(
                query="transformers", max_papers=50
            )
            assert result == "fetched 7 papers"


class TestGetStatus:
    def test_invoke_delegates_to_pipeline_status(self):
        """get_status is a thin wrapper - the actual query logic (and its
        test coverage) lives in status.py, shared with the REST endpoint."""
        fake_status = {"documents_total": 42, "embed_model": "BAAI/bge-small-en-v1.5"}
        with patch("orchestrator.tools.pipeline_status", return_value=fake_status):
            result = tools.get_status.invoke({})

        assert result == fake_status


class TestSearchChunks:
    def test_invoke_shapes_results_from_search_engine(self):
        from schemas import ChunkResult, SearchResponse

        response = SearchResponse(
            query="cervical cancer transformers",
            model="bge-small",
            reranked=True,
            results=[
                ChunkResult(
                    chunk_id=1,
                    doc_id=9,
                    title="Deep Learning for Cervical Cancer Survival",
                    authors=["A. Author"],
                    year=2022,
                    page_num=3,
                    pdf_path="x.pdf",
                    text="We used a transformer model to predict survival...",
                    score=0.87,
                )
            ],
        )
        mock_engine = MagicMock()
        mock_engine.search.return_value = response

        with patch("orchestrator.tools.get_search_engine", return_value=mock_engine):
            result = tools.search_chunks.invoke(
                {"query": "cervical cancer transformers", "top_k": 3, "rerank": True}
            )

        mock_engine.search.assert_called_once_with(
            "cervical cancer transformers", top_k=3, rerank=True, rerank_top_k=3
        )
        assert result == [
            {
                "doc_id": 9,
                "title": "Deep Learning for Cervical Cancer Survival",
                "authors": ["A. Author"],
                "year": 2022,
                "page_num": 3,
                "text": "We used a transformer model to predict survival...",
                "score": 0.87,
            }
        ]


class TestGetDocument:
    def test_invoke_returns_metadata_for_known_doc(self):
        doc = MagicMock(
            id=9,
            title="Deep Learning for Cervical Cancer Survival",
            authors=["A. Author"],
            venue="MICCAI",
            year=2022,
            abstract="An abstract.",
        )
        session = MagicMock()
        session.get.return_value = doc

        with (
            patch("orchestrator.tools.PostgresInterface.connect", return_value=MagicMock()),
            patch("orchestrator.tools.Session") as MockSession,
        ):
            MockSession.return_value.__enter__.return_value = session
            result = tools.get_document.invoke({"doc_id": 9})

        assert result == {
            "doc_id": 9,
            "title": "Deep Learning for Cervical Cancer Survival",
            "authors": ["A. Author"],
            "venue": "MICCAI",
            "year": 2022,
            "abstract": "An abstract.",
        }

    def test_invoke_returns_error_for_unknown_doc(self):
        session = MagicMock()
        session.get.return_value = None

        with (
            patch("orchestrator.tools.PostgresInterface.connect", return_value=MagicMock()),
            patch("orchestrator.tools.Session") as MockSession,
        ):
            MockSession.return_value.__enter__.return_value = session
            result = tools.get_document.invoke({"doc_id": 404})

        assert "error" in result
