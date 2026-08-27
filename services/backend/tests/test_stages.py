"""Unit tests for the per-stage entrypoints in stages/.

These mock the underlying executor classes (PdfFetcher/PdfChunker/PdfEmbedder)
and config, so they run with no DB, no Pinecone, no network - they only check
that each stage's run() wires config into the right executor call.
"""
from unittest.mock import MagicMock, patch

import pytest


def make_cfg(download=None, chunk=None, embed=None):
    cfg = MagicMock()
    cfg.stages.download.workers = 4
    cfg.stages.download.limit = 20
    cfg.stages.chunk.limit = 10
    cfg.stages.embed.limit = 500
    for k, v in (download or {}).items():
        setattr(cfg.stages.download, k, v)
    for k, v in (chunk or {}).items():
        setattr(cfg.stages.chunk, k, v)
    for k, v in (embed or {}).items():
        setattr(cfg.stages.embed, k, v)
    return cfg


class TestDownloadStage:
    def test_run_calls_fetcher_with_configured_limits(self):
        from stages import download

        cfg = make_cfg(download={"workers": 7, "limit": 42})
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

        cfg = make_cfg(chunk={"limit": 13})
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

        cfg = make_cfg(embed={"limit": 99})
        with (
            patch("stages.embed.load", return_value=cfg),
            patch("stages.embed.PdfEmbedder") as MockEmbedder,
        ):
            embed.run()

            MockEmbedder.assert_called_once_with()
            MockEmbedder.return_value.execute.assert_called_once_with(max_chunks=99)
