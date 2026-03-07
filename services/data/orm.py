import datetime
from typing import Optional

from sqlalchemy import String
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
