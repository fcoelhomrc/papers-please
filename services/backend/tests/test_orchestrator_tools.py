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


class TestDownloadPending:
    def test_invoke_calls_fetcher_with_configured_workers(self):
        cfg = MagicMock()
        cfg.stages.download.workers = 3
        with (
            patch("orchestrator.tools.load", return_value=cfg),
            patch("orchestrator.tools.PdfFetcher") as MockFetcher,
        ):
            MockFetcher.return_value.pending.return_value = [1, 2, 3, 4, 5]
            result = tools.download_pending.invoke({"limit": 2})

            MockFetcher.assert_called_once_with(max_workers=3)
            MockFetcher.return_value.execute.assert_called_once_with(limit=2)
            assert result == "attempted 2 downloads (limit=2)"


class TestChunkPending:
    def test_invoke_calls_chunker(self):
        with patch("orchestrator.tools.PdfChunker") as MockChunker:
            MockChunker.return_value.pending.return_value = [1, 2]
            result = tools.chunk_pending.invoke({"limit": 10})

            MockChunker.return_value.execute.assert_called_once_with(limit=10)
            assert result == "attempted 2 objects to chunk (limit=10)"


class TestEmbedPending:
    def test_invoke_calls_embedder(self):
        with patch("orchestrator.tools.PdfEmbedder") as MockEmbedder:
            result = tools.embed_pending.invoke({"limit": 100})

            MockEmbedder.return_value.execute.assert_called_once_with(max_chunks=100)
            assert result == "embedded up to 100 pending chunks"


class TestGetStatus:
    def test_invoke_shapes_counts_from_db(self):
        cfg = MagicMock()
        cfg.embedder.model = "bge-small"

        session = MagicMock()
        session.execute.return_value.scalar_one.side_effect = [
            42,  # documents_total
            5,   # pending_download
            3,   # chunks_pending_embed
        ]
        session.execute.return_value.all.return_value = [("pending", 2), ("chunked", 10)]

        with (
            patch("orchestrator.tools.load", return_value=cfg),
            patch("orchestrator.tools.PostgresInterface.connect", return_value=MagicMock()),
            patch("orchestrator.tools.Session") as MockSession,
        ):
            MockSession.return_value.__enter__.return_value = session
            result = tools.get_status.invoke({})

        assert result["documents_total"] == 42
        assert result["pending_download"] == 5
        assert result["objects_by_status"] == {"pending": 2, "chunked": 10}
        assert result["chunks_pending_embed"] == 3
        assert result["embed_model"] == "BAAI/bge-small-en-v1.5"
