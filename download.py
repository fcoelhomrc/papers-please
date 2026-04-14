import argparse
import logging
import os
from pathlib import Path

import logger
from dotenv import load_dotenv

from services.ingest.fetcher import PdfFetcher

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()

store_root = os.environ.get("OBJECT_STORE_ROOT", "")
Path(store_root).mkdir(parents=True, exist_ok=True)
PdfFetcher(max_workers=args.workers, store_root=store_root).execute()
