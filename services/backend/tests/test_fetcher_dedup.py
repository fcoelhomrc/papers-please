"""Tests for SemanticScholarFetcher's dedup accounting.

Unit test: mocks _write to prove fetch()'s pagination budget (max_papers)
is tracked against papers *processed*, not papers *newly inserted* - so a
mostly-duplicate query doesn't ignore max_papers and keep paginating.

Integration test: real Postgres, proves the UNIQUE constraint + returning()
count actually reports 0 new for an already-known source_id, not the
default assume-everyone-was-new count.
"""
from unittest.mock import MagicMock, patch

import pytest

from ingest.fetcher import SemanticScholarFetcher


class TestFetchNewCountAccounting:
    def test_stops_at_max_papers_processed_even_if_all_duplicates(self):
        """3 batches of 2 processed each hit max_papers=5 by the 3rd batch
        (processed: 2, 4, 6->capped at 5) even though nothing was new."""
        fetcher = SemanticScholarFetcher.__new__(SemanticScholarFetcher)
        batches = [[{"paperId": f"p{i}"} for i in range(n)] for n in (2, 2, 2)]

        with (
            patch.object(fetcher, "_paginate", return_value=iter(batches)),
            patch.object(fetcher, "_write", return_value=0) as mock_write,
        ):
            new_count = fetcher.fetch(query="x", max_papers=5)

        assert new_count == 0
        # 3 calls: batch1 (2), batch2 (2), batch3 capped to 1 (5-4=1)
        assert mock_write.call_count == 3
        assert len(mock_write.call_args_list[-1].args[0]) == 1

    def test_returns_real_new_count_not_processed_count(self):
        fetcher = SemanticScholarFetcher.__new__(SemanticScholarFetcher)
        batches = [[{"paperId": "p1"}, {"paperId": "p2"}]]

        with (
            patch.object(fetcher, "_paginate", return_value=iter(batches)),
            patch.object(fetcher, "_write", return_value=1),  # 1 of 2 was new
        ):
            new_count = fetcher.fetch(query="x", max_papers=10)

        assert new_count == 1


@pytest.mark.integration
class TestWriteDedupIntegration:
    def test_write_skips_already_known_source_id(self, configured_db):
        from sqlalchemy import insert as sa_insert
        from sqlalchemy.orm import Session

        from db.models import Document

        with Session(configured_db) as session:
            session.execute(
                sa_insert(Document).values(source_id="dup-1", title="Already known")
            )
            session.commit()

        fetcher = SemanticScholarFetcher.__new__(SemanticScholarFetcher)
        fetcher.engine = configured_db

        batch = [
            {"paperId": "dup-1", "title": "Already known (refetched)"},
            {"paperId": "new-1", "title": "Genuinely new paper"},
        ]
        inserted = fetcher._write(batch)

        assert inserted == 1  # only new-1, dup-1 silently skipped
