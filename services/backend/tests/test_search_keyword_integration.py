"""Integration test for keyword_search against real Postgres full-text search.

Run with: uv run pytest -m integration -v
"""
import pytest
from sqlalchemy import insert
from sqlalchemy.orm import Session

pytestmark = pytest.mark.integration


def _seed(engine):
    from db.models import Chunk, Document, Object

    with Session(engine) as session:
        doc_id = session.execute(
            insert(Document)
            .values(source_id="kw-doc-1", title="Test Paper")
            .returning(Document.id)
        ).scalar_one()
        obj_id = session.execute(
            insert(Object)
            .values(doc_id=doc_id, path="/dev/null", status="chunked")
            .returning(Object.id)
        ).scalar_one()
        session.execute(
            insert(Chunk).values(
                obj_id=obj_id,
                chunk_index=0,
                chunk_text="Transformers use self-attention for sequence modeling.",
                page_num=1,
            )
        )
        session.execute(
            insert(Chunk).values(
                obj_id=obj_id,
                chunk_index=1,
                chunk_text="Convolutional networks process images with local filters.",
                page_num=2,
            )
        )
        session.commit()


class TestKeywordSearchIntegration:
    def test_matches_relevant_chunk_and_ranks_it_first(self, configured_db):
        from search import keyword_search

        _seed(configured_db)

        response = keyword_search("self-attention transformers", top_k=5)

        assert response.model == "keyword"
        assert len(response.results) == 1
        assert "self-attention" in response.results[0].text

    def test_no_match_returns_empty(self, configured_db):
        from search import keyword_search

        _seed(configured_db)

        response = keyword_search("cervical cancer survival prediction", top_k=5)

        assert response.results == []

    def test_natural_language_question_matches_at_all(self, configured_db):
        """The regression that made keyword search useless: plainto_tsquery
        ANDs every lexeme, so a full question sentence needed every one of its
        words present in a single chunk. Measured 9 hits across 50 eval
        questions before the fix."""
        from search import keyword_search

        _seed(configured_db)

        response = keyword_search(
            "What do transformers use for modelling long-range dependencies?", top_k=5
        )

        assert response.results, "a question-shaped query must still match"
        assert "self-attention" in response.results[0].text

    def test_ranks_more_matched_terms_higher(self, configured_db):
        """With OR matching, the match predicate stops discriminating - so
        ts_rank has to, or every loosely-related chunk ties."""
        from search import keyword_search

        _seed(configured_db)

        response = keyword_search("self-attention transformers sequence", top_k=5)

        assert response.results
        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)

    def test_min_score_floor_can_return_nothing(self, configured_db):
        """Retrieval must be able to answer 'nothing here is relevant'."""
        from db.connection import PostgresInterface
        from sqlalchemy.orm import Session as S

        from search import _keyword_rows

        _seed(configured_db)

        with S(PostgresInterface.connect()) as session:
            unfiltered = _keyword_rows(session, "transformers", top_k=5)
            floored = _keyword_rows(session, "transformers", top_k=5, min_score=0.99)

        assert unfiltered
        assert floored == []
