import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel


class DocumentTemplate(BaseModel):
    title: str
    abstract: Optional[str] = None
    authors: Optional[str] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    publication_date: Optional[datetime.date] = None
    citation_count: Optional[int] = None
    influential_citation_count: Optional[int] = None
    s2_paper_id: str
    s2_url: Optional[str] = None
    pdf_url: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DocumentTemplate":
        authors_raw = d.get("authors")
        authors = ", ".join(a["name"] for a in authors_raw) if authors_raw else None

        pdf = d.get("openAccessPdf")
        pdf_url = pdf.get("url") if isinstance(pdf, dict) else None

        return cls(
            title=d["title"],
            abstract=d.get("abstract"),
            authors=authors,
            venue=d.get("venue"),
            year=d.get("year"),
            publication_date=d.get("publicationDate"),
            citation_count=d.get("citationCount"),
            influential_citation_count=d.get("influentialCitationCount"),
            s2_paper_id=d["paperId"],
            s2_url=d.get("url"),
            pdf_url=pdf_url,
        )
