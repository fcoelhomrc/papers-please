import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, Optional

import requests

SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

FIELDS = (
    "paperId,title,abstract,authors,venue,year,publicationDate,url,"
    "openAccessPdf,citationCount,influentialCitationCount,fieldsOfStudy,isOpenAccess"
)


def avoid_rate_limit(
    wait: float = 0.6, jitter_min: float = 0.2, jitter_max: float = 0.8
):
    time.sleep(wait + random.uniform(jitter_min, jitter_max))


def fetch_with_backoff(
    url: str, params: Dict, headers: Dict, max_retries: int = 8
) -> Dict[str, Any]:
    """Exponential backoff + full jitter retry logic for rate limits, network issues, etc."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=90)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Too Many Requests
                wait = (2**attempt) * 4 + random.uniform(0, 4)
                print(
                    f"  Rate limit (429). Sleeping {wait:.1f}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait)
                continue
            elif response.status_code >= 500:  # Server error
                wait = (2**attempt) * 2 + random.uniform(0, 2)
                print(f"  Server error {response.status_code}. Sleeping {wait:.1f}s...")
                time.sleep(wait)
                continue
            else:
                print(f"  HTTP {response.status_code}: {response.text[:400]}")
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            wait = (2**attempt) * 1.5 + random.uniform(0, 2)
            print(
                f"  Network error (attempt {attempt + 1}): {e}. Sleeping {wait:.1f}s..."
            )
            time.sleep(wait)

    raise Exception(f"Max retries ({max_retries}) exceeded for URL: {url}")


def fetch_metadata_job(
    venues: list[str], start_year: int, end_year: int | None, query: str = ""
):
    # NOTE: Default empty query fetches all papers

    if end_year is None:
        end_year = datetime.now().year

    headers = (
        {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    )

    for venue in venues:
        token: Optional[str] = None
        total_papers = 0
        page = 0

        sanitized_venue = (
            venue.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        )
        params_base = {
            "query": query,
            "venue": sanitized_venue,
            "year": f"{start_year}-{end_year}",
            "fields": FIELDS,
        }

        while True:
            page += 1
            params = params_base.copy()
            if token:
                params["token"] = token

            try:
                data = fetch_with_backoff(BASE_URL, params, headers)
            except Exception as e:
                print(f"  Failed after retries on page {page}: {e}")
                break

            papers = data.get("data", [])
            token = data.get("token")  # None when done

            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + "\n")

            total_papers += len(papers)

            if not token or len(papers) == 0:
                break

            avoid_rate_limit()


if __name__ == "__main__":
    main()
