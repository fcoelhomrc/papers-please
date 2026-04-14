from pydantic import BaseModel


class DocumentTemplate(BaseModel):
    source_id: str
    title: str
    abstract: str | None = None
    authors: list[str] | None = None
    venue: str | None = None
    year: int | None = None
    pdf_url: str | None = None

    @classmethod
    def from_s2(cls, d: dict) -> "DocumentTemplate":
        authors_raw = d.get("authors")
        pdf = d.get("openAccessPdf")
        return cls(
            source_id=d["paperId"],
            title=d["title"],
            abstract=d.get("abstract"),
            authors=[a["name"] for a in authors_raw] if authors_raw else None,
            venue=d.get("venue"),
            year=d.get("year"),
            pdf_url=pdf.get("url") or None if isinstance(pdf, dict) else None,
        )
