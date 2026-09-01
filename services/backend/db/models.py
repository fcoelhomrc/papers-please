import datetime

from sqlalchemy import ARRAY, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[str] = mapped_column(String, unique=True)
    title: Mapped[str]
    abstract: Mapped[str | None]
    authors: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    venue: Mapped[str | None]
    year: Mapped[int | None]
    pdf_url: Mapped[str | None]
    created_at: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now)


class Object(Base):
    __tablename__ = "objects"

    id: Mapped[int] = mapped_column(primary_key=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    path: Mapped[str]
    # 'pending' | 'chunked' | 'failed' | 'dead' - see schema.sql for why
    # failed and dead are separate.
    status: Mapped[str] = mapped_column(String, default="pending")
    attempts: Mapped[int] = mapped_column(default=0)
    downloaded_at: Mapped[datetime.datetime] = mapped_column(
        default=datetime.datetime.now
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    obj_id: Mapped[int] = mapped_column(ForeignKey("objects.id", ondelete="CASCADE"))
    chunk_index: Mapped[int]
    chunk_text: Mapped[str | None]
    page_num: Mapped[int | None]


class Feedback(Base):
    """A relevance judgement someone made in the UI.

    doc_id/chunk_id are plain ints, not ForeignKeys: a judgement stays true
    about a query/document pair after a re-index renumbers or removes the
    chunk it was made against, and cascading them away on every chunker
    change would defeat the point of collecting them.
    """

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String)  # 'search' | 'citation'
    query: Mapped[str]
    doc_id: Mapped[int | None]
    chunk_id: Mapped[int | None]
    verdict: Mapped[str] = mapped_column(String)  # 'up' | 'down'
    note: Mapped[str | None]
    created_at: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now)


class EmbeddingModel(Base):
    __tablename__ = "embedding_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    hf_name: Mapped[str] = mapped_column(String, unique=True)
    dims: Mapped[int]
    index_name: Mapped[str]
    created_at: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now)


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    model_id: Mapped[int] = mapped_column(
        ForeignKey("embedding_models.id", ondelete="CASCADE"), primary_key=True
    )
    embedded_at: Mapped[datetime.datetime] = mapped_column(
        default=datetime.datetime.now
    )
