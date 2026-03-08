import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from retry import retry
from sqlalchemy import create_engine, exists, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from services.data.orm import Document, Object
from services.data.templates import DocumentTemplate

SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

FIELDS = "paperId,title,abstract,authors,venue,year,publicationDate,url,openAccessPdf,citationCount,influentialCitationCount,fieldsOfStudy"


# TODO: Extract a base Fetcher class (connect, rate_limit, ...)
class SemanticScholarFetcher:
    def __init__(self):
        self.engine = self.connect()

    @staticmethod
    def connect():
        user = os.environ["POSTGRES_USER"]
        password = os.environ["POSTGRES_PASSWORD"]
        db = os.environ["POSTGRES_DB"]

        engine = create_engine(
            f"postgresql+psycopg2://{user}:{password}@localhost/{db}",
            echo=True,
        )
        return engine

    @staticmethod
    def rate_limit(wait):
        time.sleep(wait)

    @retry(tries=8, delay=1, backoff=2, jitter=(2, 6))
    def request(self, url: str, params: Dict, headers: Dict) -> Dict[str, Any]:
        response = requests.get(url, params=params, headers=headers, timeout=90)
        status_code = response.status_code
        if status_code == 200:
            return response.json()
        else:
            raise requests.HTTPError(status_code)

    def sanitize(self, text: str) -> str:
        return (
            text.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        )

    def write_documents(self, documents: list[Dict[str, Any]]):
        rows = [DocumentTemplate.from_dict(d).model_dump() for d in documents]
        with Session(self.engine) as session:
            session.execute(
                insert(Document).on_conflict_do_nothing(index_elements=["s2_paper_id"]),
                rows,
            )
            session.commit()

    def request_batch(
        self, venues: list[str], start_year: int, end_year: int | None, query: str = ""
    ) -> None:
        if end_year is None:
            end_year = datetime.now().year

        headers = (
            {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
        )

        for venue in venues:
            token: Optional[str] = None
            params_base = {
                "query": query,
                "venue": self.sanitize(venue),
                "year": f"{start_year}-{end_year}",
                "fields": FIELDS,
            }

            while True:
                params = params_base.copy()
                if token:
                    params["token"] = token

                try:
                    data = self.request(url=BASE_URL, params=params, headers=headers)
                except Exception:
                    break

                documents = data.get("data", [])
                token = data.get("token")  # None when done
                self.write_documents(documents)
                if not token or len(documents) == 0:
                    break
                self.rate_limit(1)


class PdfFetcher:
    def __init__(self, max_workers: int, store_root: str):
        self.max_workers = max_workers
        self.store_root = store_root

        self.engine = self.connect()

    @staticmethod
    def connect():
        user = os.environ["POSTGRES_USER"]
        password = os.environ["POSTGRES_PASSWORD"]
        db = os.environ["POSTGRES_DB"]

        engine = create_engine(
            f"postgresql+psycopg2://{user}:{password}@localhost/{db}",
            echo=True,
        )
        return engine

    @staticmethod
    def rate_limit(wait):
        time.sleep(wait)

    def query(self) -> list[tuple[int, str, Path]]:
        stmt = (
            select(Document.id, Document.s2_paper_id, Document.pdf_url)
            .where(Document.pdf_url.is_not(None))
            .where(~exists().where(Object.doc_id == Document.id))
        )
        with Session(self.engine) as session:
            rows = session.execute(stmt).all()

        params = []
        for row in rows:
            doc_id = row.id
            url = row.pdf_url
            path = Path(self.store_root) / f"{row.s2_paper_id}.pdf"
            params.append((doc_id, url, path))
        print(f"Found {len(params)} records to process...")
        return params

    @staticmethod
    @retry(tries=3, delay=1, backoff=2)
    def download(url: str) -> bytes:
        print(f"Downloading {url}...")
        try:
            response = requests.get(url, timeout=90)
            status_code = response.status_code
        except Exception as error:
            raise error

        if status_code == 200:
            return response.content
        else:
            raise requests.HTTPError(status_code)

    @staticmethod
    def write(content: bytes, path: Path):
        print(f"Saving to {str(path)}...")
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(content)
        tmp.rename(path)

    def register(self, doc_id: int, path: Path):
        print(f"Writting record...")
        obj = Object(doc_id=doc_id, path=str(path))
        with Session(self.engine) as session:
            session.add(obj)
            session.commit()

    def process(self, doc_id, url, path):
        try:
            content = self.download(url)
            self.write(content, path)
            self.register(doc_id, path)
        except Exception as error:
            print(f"{error} -> ({doc_id}, {url}, {path})")

    def execute(self):
        params = self.query()
        # for p in params:  # doc_id, url, path
        #     print(p)
        #     self.process(*p)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(lambda p: self.process(*p), params)
