"""Tests for eval/seed.py - ensure_fixtures_seeded() makes eval reproducible
regardless of the dev DB's current state.

Doesn't import ragas (unlike eval/run.py), so no isolation needed here.
"""
from unittest.mock import MagicMock, patch

import pytest

from eval.seed import ensure_fixtures_seeded


class TestEnsureFixturesSeededUnit:
    def test_skips_entirely_when_all_fixtures_already_present(self):
        session = MagicMock()
        session.execute.return_value.scalars.return_value.all.return_value = [
            "eval-fixture-icub-promp",
            "eval-fixture-muscle-synergy-rl",
            "eval-fixture-fall-recovery-wheeled",
            "eval-fixture-supervisory-control-review",
        ]

        with (
            patch("eval.seed.PostgresInterface.connect", return_value=MagicMock()),
            patch("eval.seed.Session") as MockSession,
            patch("eval.seed.PdfEmbedder") as MockEmbedder,
        ):
            MockSession.return_value.__enter__.return_value = session
            ensure_fixtures_seeded()

            MockEmbedder.assert_not_called()

    def test_inserts_missing_fixtures_and_embeds(self):
        session = MagicMock()
        session.execute.return_value.scalars.return_value.all.return_value = []  # none present
        # every insert(...).returning(...) call needs a scalar_one() result
        session.execute.return_value.scalar_one.return_value = 1

        with (
            patch("eval.seed.PostgresInterface.connect", return_value=MagicMock()),
            patch("eval.seed.Session") as MockSession,
            patch("eval.seed.PdfEmbedder") as MockEmbedder,
        ):
            MockSession.return_value.__enter__.return_value = session
            ensure_fixtures_seeded()

            MockEmbedder.return_value.execute.assert_called_once()


@pytest.mark.integration
class TestEnsureFixturesSeededIntegration:
    def test_seeds_real_rows_and_is_idempotent(self, configured_db):
        """Real Postgres - proves the actual SQL inserts correct rows and
        that a second call doesn't duplicate them. Embedding itself
        (PdfEmbedder) is mocked here - that logic already has its own
        integration test (test_embedder_integration.py); this test is
        about the seed/skip decision, not re-proving embedding works."""
        from sqlalchemy import select
        from sqlalchemy.orm import Session

        from db.models import Chunk, Document
        from eval.fixtures import FIXTURES

        with patch("eval.seed.PdfEmbedder") as MockEmbedder:
            ensure_fixtures_seeded()
            ensure_fixtures_seeded()  # second call: should be a no-op

        assert MockEmbedder.return_value.execute.call_count == 1

        with Session(configured_db) as session:
            docs = session.execute(
                select(Document.source_id).where(
                    Document.source_id.in_([f["source_id"] for f in FIXTURES])
                )
            ).scalars().all()
            assert sorted(docs) == sorted(f["source_id"] for f in FIXTURES)

            chunk_count = len(
                session.execute(select(Chunk.id)).scalars().all()
            )
            assert chunk_count == sum(len(f["chunks"]) for f in FIXTURES)
