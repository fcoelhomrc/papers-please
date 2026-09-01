import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from retry import retry
from sqlalchemy import exists, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from db.connection import PostgresInterface
from db.models import Document, Object
from ingest.schemas import DocumentTemplate

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "paperId,title,abstract,authors,venue,year,openAccessPdf"

MIN_PDF_BYTES = 1024
PDF_MAGIC = b"%PDF"


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

    def _write(self, documents: list[dict]) -> int:
        """Insert, skipping documents whose source_id already exists (dedup
        guard - source_id is UNIQUE). Returns the count actually inserted,
        not len(documents), so callers can tell real new papers from
        duplicates silently skipped."""
        rows = [DocumentTemplate.from_s2(d).model_dump() for d in documents]
        with Session(self.engine) as session:
            result = session.execute(
                insert(Document)
                .on_conflict_do_nothing(index_elements=["source_id"])
                .returning(Document.id),
                rows,
            )
            inserted = len(result.fetchall())
            session.commit()
        return inserted

    def fetch(
        self,
        query: str = "",
        venue: str | None = None,
        year: str | None = None,
        max_papers: int = 500,
    ) -> int:
        """Returns the count of genuinely new papers added - not the count
        of API results processed. Re-fetching an already-known query returns
        0, rather than reporting max_papers as if all of them were new."""
        params = {"fields": FIELDS, "limit": 1000}
        if query:
            params["query"] = query
        if venue:
            params["venue"] = venue
        if year:
            params["year"] = year

        processed = 0
        new_count = 0
        for batch in self._paginate(params):
            batch = batch[: max_papers - processed]
            if batch:
                new_count += self._write(batch)
                processed += len(batch)
            if processed >= max_papers:
                break
        skipped = processed - new_count
        logger.info(
            f"Fetched {new_count} new papers ({skipped} already known, "
            f"venue={venue}, query={query})"
        )
        return new_count


class PdfFetcher(PostgresInterface):
    def __init__(self, max_workers: int, store_root: str | None = None):
        from config import load
        super().__init__()
        self.max_workers = max_workers
        self.max_attempts = load().stages.download.max_attempts
        self.store_root = Path(store_root or load().storage.root)
        self._tmp_dir = self.store_root / ".tmp"

    def pending(self) -> list[tuple[int, str, str]]:
        """Documents still worth attempting a download for.

        Either never attempted (no objects row) or attempted and still under
        the cap. A row that reached the cap is 'failed' and is deliberately
        not re-selected - that is the whole point of recording the attempt.

        Ordered by id: without an ORDER BY, Postgres may return rows in any
        order, and `execute(limit=...)` then slices an arbitrary subset -
        so a document could be starved indefinitely while others are
        retried. Oldest-first also means a backlog drains in the order it
        arrived.
        """
        retryable = exists().where(
            (Object.doc_id == Document.id)
            & (Object.status == "downloading")
            & (Object.attempts < self.max_attempts)
        )
        stmt = (
            select(Document.id, Document.source_id, Document.pdf_url)
            .where(Document.pdf_url.is_not(None))
            .where(Document.pdf_url != "")
            .where((~exists().where(Object.doc_id == Document.id)) | retryable)
            .order_by(Document.id)
        )
        with Session(self.engine) as session:
            rows = session.execute(stmt).all()
        result = [(r.id, r.pdf_url, f"{r.source_id}.pdf") for r in rows]
        logger.info(f"{len(result)} PDFs pending")
        return result

    @staticmethod
    @retry(tries=3, delay=1, backoff=2)
    def download(url: str) -> bytes:
        """Fetch a PDF, or raise.

        A 200 is not enough. Many open-access links resolve to an HTML
        landing page, and some publishers serve one with a 200 rather than
        an error - so the old status-code-only check would write the page to
        `<source_id>.pdf`, register it as a real object, and hand it to
        Docling, which then burned the chunker's whole retry budget on OCR
        of a web page.

        Checked here rather than in a sweep afterwards: this is the boundary
        where bad bytes enter the system, and rejecting them costs one `if`
        while cleaning them up later costs a reconciliation pass.
        """
        response = requests.get(url, timeout=90)
        if response.status_code != 200:
            raise requests.HTTPError(response.status_code)

        # The magic bytes decide, not the header: servers lie in both
        # directions - a real PDF is routinely served as octet-stream, and a
        # landing page is sometimes labelled application/pdf. The header is
        # only used to make the error message useful.
        if not response.content.startswith(PDF_MAGIC):
            content_type = response.headers.get("content-type", "?").split(";")[0].strip()
            raise ValueError(f"not a PDF (no %PDF header; content-type: {content_type})")
        if len(response.content) < MIN_PDF_BYTES:
            raise ValueError(f"truncated PDF ({len(response.content)} bytes)")
        return response.content

    def save(self, content: bytes, path: Path):
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._tmp_dir / path.name
        tmp.write_bytes(content)
        tmp.rename(path)  # atomic on same filesystem

    def _begin_attempt(self, doc_id: int, filename: str):
        """Record the attempt *before* the request goes out.

        Counting after the fact loses the attempt whenever the process dies
        mid-download - and an uncounted attempt is an infinite retry with
        extra steps, which is exactly the bug this replaces.
        """
        with Session(self.engine) as session:
            obj = session.execute(
                select(Object).where(Object.doc_id == doc_id)
            ).scalar_one_or_none()
            if obj is None:
                session.add(
                    Object(
                        doc_id=doc_id, path=filename, status="downloading", attempts=1
                    )
                )
            else:
                obj.status = "downloading"
                obj.attempts += 1
                obj.path = filename
            session.commit()

    def _finish_attempt(self, doc_id: int, ok: bool):
        """Move the row out of 'downloading', or give up on it.

        'pending' is the chunker's entry condition, so a successful download
        hands the object straight to the next stage. A failure stays
        'downloading' - meaning "will be retried" - until the cap is
        reached, at which point 'failed' takes it out of pending() for good.
        """
        with Session(self.engine) as session:
            obj = session.execute(
                select(Object).where(Object.doc_id == doc_id)
            ).scalar_one_or_none()
            if obj is None:
                return
            if ok:
                obj.status = "pending"
            elif obj.attempts >= self.max_attempts:
                obj.status = "failed"
                logger.warning(
                    f"Giving up on doc {doc_id} after {obj.attempts} download attempts"
                )
            session.commit()

    def task(self, doc_id: int, url: str, filename: str) -> bool:
        path = self.store_root / filename
        self._begin_attempt(doc_id, filename)
        try:
            content = self.download(url)
            self.save(content, path)
        except Exception as e:
            self._finish_attempt(doc_id, ok=False)
            logger.error(f"Failed ({filename}): {e}")
            return False
        self._finish_attempt(doc_id, ok=True)
        logger.info(f"Downloaded {filename} ({len(content) / 1024:.0f} KB)")
        return True

    def execute(self, limit: int | None = None):
        pending = self.pending()
        if limit:
            pending = pending[:limit]
        if not pending:
            logger.info("Nothing to download")
            return
        done = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.task, *p): p[2] for p in pending}
            for future in as_completed(futures):
                if future.result():
                    done += 1
                logger.info(f"Progress: {done}/{len(pending)}")
