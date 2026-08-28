"""Integration test for GET /documents filtering (has_pdf/processed flags,
only_available/only_processed filters) against real Postgres - the
exists()-subquery SQL is exactly the kind of thing a mock would happily
validate even if backwards.
"""
import pytest
from sqlalchemy import insert
from sqlalchemy.orm import Session
from starlette.testclient import TestClient

pytestmark = pytest.mark.integration


def _seed(engine):
    from db.models import Chunk, ChunkEmbedding, Document, EmbeddingModel, Object

    with Session(engine) as session:
        # doc A: no PDF at all
        doc_a = session.execute(
            insert(Document).values(source_id="a", title="No PDF Paper").returning(Document.id)
        ).scalar_one()

        # doc B: has PDF, not yet chunked/embedded
        doc_b = session.execute(
            insert(Document).values(source_id="b", title="Downloaded Only Paper").returning(Document.id)
        ).scalar_one()
        session.execute(insert(Object).values(doc_id=doc_b, path="b.pdf", status="pending"))

        # doc C: has PDF, fully processed (chunked + embedded)
        doc_c = session.execute(
            insert(Document).values(source_id="c", title="Fully Processed Paper").returning(Document.id)
        ).scalar_one()
        obj_c = session.execute(
            insert(Object).values(doc_id=doc_c, path="c.pdf", status="chunked").returning(Object.id)
        ).scalar_one()
        chunk_c = session.execute(
            insert(Chunk)
            .values(obj_id=obj_c, chunk_index=0, chunk_text="hello", page_num=1)
            .returning(Chunk.id)
        ).scalar_one()
        model_id = session.execute(
            insert(EmbeddingModel)
            .values(hf_name="test-model", dims=4, index_name="test-index")
            .returning(EmbeddingModel.id)
        ).scalar_one()
        session.execute(insert(ChunkEmbedding).values(chunk_id=chunk_c, model_id=model_id))
        session.commit()

    return {"a": doc_a, "b": doc_b, "c": doc_c}


class TestDocumentsFilterIntegration:
    def test_has_pdf_and_processed_flags_are_correct(self, configured_db, monkeypatch):
        doc_ids = _seed(configured_db)

        import api
        from search import SearchEngine

        fake_engine = SearchEngine.__new__(SearchEngine)
        fake_engine.engine = configured_db
        monkeypatch.setattr(api, "get_search_engine", lambda: fake_engine)

        client = TestClient(api.app)
        response = client.get("/documents", params={"sort": "title"})
        docs = {d["source_id"]: d for d in response.json()}

        assert docs["a"]["has_pdf"] is False
        assert docs["a"]["processed"] is False
        assert docs["b"]["has_pdf"] is True
        assert docs["b"]["processed"] is False
        assert docs["c"]["has_pdf"] is True
        assert docs["c"]["processed"] is True

    def test_only_processed_filter_returns_just_the_processed_doc(self, configured_db, monkeypatch):
        _seed(configured_db)

        import api
        from search import SearchEngine

        fake_engine = SearchEngine.__new__(SearchEngine)
        fake_engine.engine = configured_db
        monkeypatch.setattr(api, "get_search_engine", lambda: fake_engine)

        client = TestClient(api.app)
        response = client.get("/documents", params={"only_processed": True})
        docs = response.json()

        assert len(docs) == 1
        assert docs[0]["source_id"] == "c"
