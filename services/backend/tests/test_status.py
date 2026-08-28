"""Unit tests for status.pipeline_status() - the query logic shared between
the REST /status endpoint and the orchestrator's get_status tool.
"""
from unittest.mock import MagicMock, patch

import status


class TestPipelineStatus:
    def test_shapes_counts_from_db(self):
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
            patch("status.load", return_value=cfg),
            patch("status.PostgresInterface.connect", return_value=MagicMock()),
            patch("status.Session") as MockSession,
        ):
            MockSession.return_value.__enter__.return_value = session
            result = status.pipeline_status()

        assert result["documents_total"] == 42
        assert result["pending_download"] == 5
        assert result["objects_by_status"] == {"pending": 2, "chunked": 10}
        assert result["chunks_pending_embed"] == 3
        assert result["embed_model"] == "BAAI/bge-small-en-v1.5"
