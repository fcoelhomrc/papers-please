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
