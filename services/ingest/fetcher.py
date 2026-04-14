import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests
from retry import retry
from sqlalchemy import exists, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from db.connection import PostgresInterface
from db.models import Document, Object
from services.ingest.schemas import DocumentTemplate

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "paperId,title,abstract,authors,venue,year,openAccessPdf"


class SemanticScholarFetcher(PostgresInterface):
    def __init__(self):
        super().__init__()
        self._headers = (
            {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
        )

    @retry(tries=8, delay=1, backoff=2, jitter=(2, 6))
    def _get(self, params: dict) -> dict:
        response = requests.get(
            BASE_URL, params=params, headers=self._headers, timeout=90
        )
        if response.status_code == 200:
            return response.json()
        raise requests.HTTPError(response.status_code)

    def _paginate(self, params: dict) -> Iterator[list[dict]]:
        token = None
        while True:
            if token:
                params = {**params, "token": token}
            data = self._get(params)
            batch = data.get("data", [])
            token = data.get("token")
            yield batch
            if not token or not batch:
                break
            self.rate_limit(1)

    def _write(self, documents: list[dict]):
        rows = [DocumentTemplate.from_s2(d).model_dump() for d in documents]
        with Session(self.engine) as session:
            session.execute(
                insert(Document).on_conflict_do_nothing(index_elements=["source_id"]),
                rows,
            )
            session.commit()

    def fetch(
        self,
        query: str = "",
        venue: str | None = None,
        year: str | None = None,
        max_papers: int = 500,
    ) -> int:
        params = {"fields": FIELDS, "limit": 1000}
        if query:
            params["query"] = query
        if venue:
            params["venue"] = venue
        if year:
            params["year"] = year

        total = 0
        for batch in self._paginate(params):
            batch = batch[: max_papers - total]
            if batch:
                self._write(batch)
                total += len(batch)
            if total >= max_papers:
                break
        logger.info(f"Fetched {total} papers (venue={venue}, query={query})")
        return total


class PdfFetcher(PostgresInterface):
    def __init__(self, max_workers: int, store_root: str):
        super().__init__()
        self.max_workers = max_workers
        self.store_root = store_root

    def pending(self) -> list[tuple[int, str, str]]:
        stmt = (
            select(Document.id, Document.source_id, Document.pdf_url)
            .where(Document.pdf_url.is_not(None))
            .where(Document.pdf_url != "")
            .where(~exists().where(Object.doc_id == Document.id))
        )
        with Session(self.engine) as session:
            rows = session.execute(stmt).all()
        result = [
            (r.id, r.pdf_url, f"{r.source_id}.pdf")
            for r in rows
        ]
        logger.info(f"{len(result)} PDFs pending")
        return result

    @staticmethod
    @retry(tries=3, delay=1, backoff=2)
    def download(url: str) -> bytes:
        response = requests.get(url, timeout=90)
        if response.status_code == 200:
            return response.content
        raise requests.HTTPError(response.status_code)

    @staticmethod
    def save(content: bytes, path: Path):
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(content)
        tmp.rename(path)

    def register(self, doc_id: int, filename: str):
        with Session(self.engine) as session:
            session.add(Object(doc_id=doc_id, path=filename))
            session.commit()

    def task(self, doc_id: int, url: str, filename: str):
        path = Path(self.store_root) / filename
        try:
            self.save(self.download(url), path)
            self.register(doc_id, filename)
        except Exception as e:
            logger.error(f"Failed ({doc_id}, {url}): {e}")

    def execute(self):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(lambda p: self.task(*p), self.pending())
