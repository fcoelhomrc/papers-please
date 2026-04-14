import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from retry import retry
from sqlalchemy import delete, exists, select
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
        self.store_root = Path(store_root)
        self._tmp_dir = self.store_root / ".tmp"

    def pending(self) -> list[tuple[int, str, str]]:
        stmt = (
            select(Document.id, Document.source_id, Document.pdf_url)
            .where(Document.pdf_url.is_not(None))
            .where(Document.pdf_url != "")
            .where(~exists().where(Object.doc_id == Document.id))
        )
        with Session(self.engine) as session:
            rows = session.execute(stmt).all()
        result = [(r.id, r.pdf_url, f"{r.source_id}.pdf") for r in rows]
        logger.info(f"{len(result)} PDFs pending")
        return result

    @staticmethod
    @retry(tries=3, delay=1, backoff=2)
    def download(url: str) -> bytes:
        response = requests.get(url, timeout=90)
        if response.status_code == 200:
            return response.content
        raise requests.HTTPError(response.status_code)

    def save(self, content: bytes, path: Path):
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._tmp_dir / path.name
        tmp.write_bytes(content)
        tmp.rename(path)  # atomic on same filesystem

    def register(self, doc_id: int, filename: str):
        with Session(self.engine) as session:
            session.add(Object(doc_id=doc_id, path=filename))
            session.commit()

    def task(self, doc_id: int, url: str, filename: str) -> bool:
        path = self.store_root / filename
        try:
            content = self.download(url)
            self.save(content, path)
            self.register(doc_id, filename)
            logger.info(f"Downloaded {filename} ({len(content) / 1024:.0f} KB)")
            return True
        except Exception as e:
            logger.error(f"Failed ({filename}): {e}")
            return False

    def execute(self):
        pending = self.pending()
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

    def reconcile(self):
        store = self.store_root

        # Remove leftover tmp files from interrupted downloads
        if self._tmp_dir.exists():
            for f in self._tmp_dir.iterdir():
                f.unlink()
                logger.warning(f"Removed incomplete download: {f.name}")

        # Load registered objects from DB
        with Session(self.engine) as session:
            registered = {
                r.path: r.id
                for r in session.execute(select(Object.path, Object.id)).all()
            }

        # Check each registered file for existence / corruption
        corrupt_ids = []
        for filename, obj_id in registered.items():
            path = store / filename
            reason = None
            if not path.exists():
                reason = "missing"
            elif path.stat().st_size < MIN_PDF_BYTES:
                reason = f"too small ({path.stat().st_size} bytes)"
                path.unlink()
            else:
                with open(path, "rb") as f:
                    magic = f.read(4)
                if magic != PDF_MAGIC:
                    reason = "not a valid PDF"
                    path.unlink()
            if reason:
                logger.warning(f"Corrupt object {obj_id} ({filename}): {reason} — will re-download")
                corrupt_ids.append(obj_id)

        if corrupt_ids:
            with Session(self.engine) as session:
                session.execute(delete(Object).where(Object.id.in_(corrupt_ids)))
                session.commit()

        # Remove stray PDFs not registered in DB
        registered_names = set(registered.keys())
        stray = 0
        for f in store.iterdir():
            if f.is_file() and f.suffix == ".pdf" and f.name not in registered_names:
                f.unlink()
                stray += 1
                logger.warning(f"Removed stray file: {f.name}")

        logger.info(f"Reconcile done — {len(corrupt_ids)} corrupt, {stray} stray removed")
