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


class TestQueueItems:
    """#35 — the Queue page showed aggregate counts only, so a stuck document
    was indistinguishable from a busy pipeline."""

    def _session(self, rows, chunks=None, embedded=None):
        from unittest.mock import MagicMock

        session = MagicMock()
        session.__enter__ = lambda s: s
        session.__exit__ = lambda s, *a: None

        results = [MagicMock(all=MagicMock(return_value=rows))]
        if any(r.obj_id is not None for r in rows):
            results.append(MagicMock(all=MagicMock(return_value=list((chunks or {}).items()))))
            results.append(MagicMock(all=MagicMock(return_value=list((embedded or {}).items()))))
        session.execute.side_effect = results
        return session

    def _row(self, doc_id, title="T", pdf_url="http://x/y.pdf", obj_id=1, status="pending", attempts=0):
        from types import SimpleNamespace

        return SimpleNamespace(
            id=doc_id, title=title, pdf_url=pdf_url, obj_id=obj_id, status=status, attempts=attempts
        )

    def _run(self, rows, chunks=None, embedded=None):
        from unittest.mock import patch

        import status as status_module

        with (
            patch.object(status_module, "Session", return_value=self._session(rows, chunks, embedded)),
            patch.object(status_module.PostgresInterface, "connect", return_value=None),
        ):
            return status_module.queue_items()

    def test_a_fully_embedded_paper_reads_as_done(self):
        """'chunked' is where the objects table stops; whether the chunks
        reached the index lives in chunk_embeddings, and done is the state a
        reader wants to see."""
        items = self._run(
            [self._row(1, obj_id=10, status="chunked")], chunks={10: 5}, embedded={10: 5}
        )

        assert items[0]["status"] == "embedded"

    def test_partially_embedded_is_still_chunked(self):
        items = self._run(
            [self._row(1, obj_id=10, status="chunked")], chunks={10: 5}, embedded={10: 2}
        )

        assert items[0]["status"] == "chunked"
        assert (items[0]["chunks"], items[0]["embedded"]) == (5, 2)

    def test_a_paper_with_no_pdf_is_not_queued_for_anything(self):
        """Listing metadata-only papers as awaiting download would make a
        queue that never drains."""
        items = self._run([self._row(1, pdf_url=None, obj_id=None)])

        assert items[0]["status"] == "metadata_only"

    def test_a_paper_with_a_pdf_and_no_object_is_awaiting_download(self):
        items = self._run([self._row(1, obj_id=None)])

        assert items[0]["status"] == "awaiting_download"

    def test_attention_first_ordering(self):
        """Sorting by id or timestamp would bury a dead object under
        hundreds of completed ones, which is the opposite of what a queue
        view is for."""
        rows = [
            self._row(1, obj_id=10, status="chunked"),
            self._row(2, obj_id=20, status="failed", attempts=1),
            self._row(3, obj_id=30, status="dead", attempts=3),
            self._row(4, obj_id=40, status="pending"),
        ]

        items = self._run(rows, chunks={10: 1}, embedded={10: 1})

        assert [i["status"] for i in items] == ["dead", "failed", "pending", "embedded"]

    def test_attempts_are_reported(self):
        items = self._run([self._row(1, obj_id=10, status="dead", attempts=3)])

        assert items[0]["attempts"] == 3
