import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import requests
from retry import retry
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from services.data.orm import Document
from services.data.templates import DocumentTemplate

SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

FIELDS = "paperId,title,abstract,authors,venue,year,publicationDate,url,openAccessPdf,citationCount,influentialCitationCount,fieldsOfStudy"


class Fetcher:
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

    def fetch_batch(
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
