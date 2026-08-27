"""Integration test for PdfEmbedder against real Postgres + real Pinecone.

Run with: uv run pytest -m integration -v
Needs: podman on PATH, network access, PINECONE_API_KEY in the environment.
"""
import pytest
from sqlalchemy import insert
from sqlalchemy.orm import Session

pytestmark = pytest.mark.integration


def _seed_one_chunk(engine, text="Attention is all you need."):
    from db.models import Chunk, Document, Object

    with Session(engine) as session:
        doc_id = session.execute(
            insert(Document)
            .values(source_id="test-doc-1", title="Test Paper")
            .returning(Document.id)
        ).scalar_one()
        obj_id = session.execute(
            insert(Object)
            .values(doc_id=doc_id, path="/dev/null", status="chunked")
            .returning(Object.id)
        ).scalar_one()
        chunk_id = session.execute(
            insert(Chunk)
            .values(obj_id=obj_id, chunk_index=0, chunk_text=text, page_num=1)
            .returning(Chunk.id)
        ).scalar_one()
        session.commit()
    return chunk_id


class TestPdfEmbedderIntegration:
    def test_execute_embeds_pending_chunk_into_postgres_and_pinecone(
        self, configured_db, pinecone_index
    ):
        from db.models import ChunkEmbedding
        from process.embedder import PdfEmbedder

        chunk_id = _seed_one_chunk(configured_db)

        embedder = PdfEmbedder(model_key="bge-small", namespace="test")
        embedder.execute()

        # Postgres: chunk is recorded as embedded.
        with Session(configured_db) as session:
            recorded = session.query(ChunkEmbedding).filter_by(chunk_id=chunk_id).all()
        assert len(recorded) == 1

        # Pinecone: the vector actually landed in the test namespace.
        stats = pinecone_index.describe_index_stats()
        assert stats.namespaces.get("test", {}).get("vector_count", 0) >= 1

        fetched = pinecone_index.fetch(ids=[str(chunk_id)], namespace="test")
        assert str(chunk_id) in fetched.vectors
