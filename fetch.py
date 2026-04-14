import argparse
import logging
import os

import logger
from dotenv import load_dotenv

from services.ingest.fetcher import SemanticScholarFetcher

load_dotenv()

DEFAULT_VENUES = [
    "CVPR", "ICCV", "ECCV",
    "NeurIPS", "ICML", "ICLR", "AAAI",
    "MICCAI", "MIDL", "ISBI",
    "Medical Image Analysis",
    "IEEE Transactions on Medical Imaging",
    "Nature", "Nature Medicine", "Nature Cancer",
    "Nature Methods", "Nature Communications",
    "Nature Biomedical Engineering",
    "npj Precision Oncology", "npj Digital Medicine", "npj Genomic Medicine",
    "Lancet Oncology", "Cancer Research", "Journal of Clinical Oncology",
    "Radiology", "Radiology: Artificial Intelligence",
]

parser = argparse.ArgumentParser()
parser.add_argument("--venue", action="append", dest="venues", metavar="VENUE")
parser.add_argument("--query", default="")
parser.add_argument("--year", default=None, help="e.g. 2020-2025")
parser.add_argument("--max-papers", type=int, default=500)
args = parser.parse_args()

venues = args.venues or DEFAULT_VENUES
fetcher = SemanticScholarFetcher()
for venue in venues:
    fetcher.fetch(query=args.query, venue=venue, year=args.year, max_papers=args.max_papers)
