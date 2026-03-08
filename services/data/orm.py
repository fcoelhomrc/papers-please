import datetime
from typing import Optional

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str]
    abstract: Mapped[Optional[str]]
    authors: Mapped[Optional[str]]
    venue: Mapped[Optional[str]]
    year: Mapped[Optional[int]]
    publication_date: Mapped[Optional[datetime.date]]
    citation_count: Mapped[Optional[int]]
    influential_citation_count: Mapped[Optional[int]]
    s2_paper_id: Mapped[str] = mapped_column(String, unique=True)
    s2_url: Mapped[Optional[str]]
    pdf_url: Mapped[Optional[str]]


class Object(Base):
    __tablename__ = "objects"

    id: Mapped[int] = mapped_column(primary_key=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    path: Mapped[str]
    chunk_status: Mapped[str] = mapped_column(String, default="pending")
