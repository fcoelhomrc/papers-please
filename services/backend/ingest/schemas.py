from pydantic import BaseModel

ARXIV_PDF = "https://arxiv.org/pdf/{}"


def arxiv_pdf_url(d: dict) -> str | None:
    """A direct PDF URL, or None if this paper has no arXiv id.

    Synthesised from the id rather than read from `openAccessPdf.url`,
    which for arXiv papers usually points somewhere else entirely - ACL
    Anthology, MDPI, or a doi.org landing page that answers 403 to a
    scripted fetch. In one sampled page of 1000 results, 187 papers had an
    arXiv id and only 73 had an arxiv.org PDF URL.
    """
    arxiv_id = (d.get("externalIds") or {}).get("ArXiv")
    return ARXIV_PDF.format(arxiv_id) if arxiv_id else None


class DocumentTemplate(BaseModel):
    source_id: str
    title: str
    abstract: str | None = None
    authors: list[str] | None = None
    venue: str | None = None
    year: int | None = None
    pdf_url: str | None = None
    citation_count: int | None = None
    corpus: str = "main"
    topic: str | None = None

    @classmethod
    def from_s2(
        cls, d: dict, corpus: str = "main", topic: str | None = None
    ) -> "DocumentTemplate":
        authors_raw = d.get("authors")
        pdf = d.get("openAccessPdf")
        fallback = pdf.get("url") or None if isinstance(pdf, dict) else None
        return cls(
            source_id=d["paperId"],
            title=d["title"],
            abstract=d.get("abstract"),
            authors=[a["name"] for a in authors_raw] if authors_raw else None,
            venue=d.get("venue"),
            year=d.get("year"),
            # arXiv wins when available - it is a file, the alternative is
            # usually a landing page.
            pdf_url=arxiv_pdf_url(d) or fallback,
            citation_count=d.get("citationCount"),
            corpus=corpus,
            topic=topic,
        )
