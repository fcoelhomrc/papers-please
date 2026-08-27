"""Unit tests for the per-stage entrypoints in stages/.

These mock the underlying executor classes (PdfFetcher/PdfChunker/PdfEmbedder)
and config, so they run with no DB, no Pinecone, no network - they only check
that each stage's run() wires config into the right executor call.
"""
from unittest.mock import MagicMock, patch

import pytest


def make_cfg(**worker_overrides):
    cfg = MagicMock()
    cfg.worker.download_workers = 4
    cfg.worker.download_limit = 20
    cfg.worker.chunk_limit = 10
    cfg.worker.embed_limit = 500
    for k, v in worker_overrides.items():
        setattr(cfg.worker, k, v)
    return cfg


class TestDownloadStage:
    def test_run_calls_fetcher_with_configured_limits(self):
        from stages import download

        cfg = make_cfg(download_workers=7, download_limit=42)
        with (
            patch("stages.download.load", return_value=cfg),
            patch("stages.download.PdfFetcher") as MockFetcher,
        ):
            download.run()

            MockFetcher.assert_called_once_with(max_workers=7)
            MockFetcher.return_value.execute.assert_called_once_with(limit=42)


class TestChunkStage:
    def test_run_calls_chunker_with_configured_limit(self):
        from stages import chunk

        cfg = make_cfg(chunk_limit=13)
        with (
            patch("stages.chunk.load", return_value=cfg),
            patch("stages.chunk.PdfChunker") as MockChunker,
        ):
            chunk.run()

            MockChunker.assert_called_once_with()
            MockChunker.return_value.execute.assert_called_once_with(limit=13)


class TestEmbedStage:
    def test_run_calls_embedder_with_configured_max_chunks(self):
        from stages import embed

        cfg = make_cfg(embed_limit=99)
        with (
            patch("stages.embed.load", return_value=cfg),
            patch("stages.embed.PdfEmbedder") as MockEmbedder,
        ):
            embed.run()

            MockEmbedder.assert_called_once_with()
            MockEmbedder.return_value.execute.assert_called_once_with(max_chunks=99)
