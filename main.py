import os
from pathlib import Path
from dotenv import load_dotenv

from services.data.fetcher import PdfFetcher, SemanticScholarFetcher


load_dotenv()


# FIXME: Handle this properly e.g. at Docker Compose
OBJECT_STORE_ROOT = os.environ.get("OBJECT_STORE_ROOT", "")
Path(OBJECT_STORE_ROOT).mkdir(parents=True, exist_ok=True)

fetcher = SemanticScholarFetcher()
downloader = PdfFetcher(max_workers=8, store_root=OBJECT_STORE_ROOT)

venues = [
    # Computer Vision
    "CVPR",
    "ICCV",
    "ECCV",
    # Machine Learning
    "NeurIPS",
    "ICML",
    "ICLR",
    "AAAI",
    # Medical Imaging conferences
    "MICCAI",
    "MIDL",
    "ISBI",
    # Medical Imaging journals
    "Medical Image Analysis",
    "IEEE Transactions on Medical Imaging",
    # Nature family
    "Nature",
    "Nature Medicine",
    "Nature Cancer",
    "Nature Methods",
    "Nature Communications",
    "Nature Biomedical Engineering",
    # npj journals
    "npj Precision Oncology",
    "npj Digital Medicine",
    "npj Genomic Medicine",
    # Clinical/oncology journals
    "Lancet Oncology",
    "Cancer Research",
    "Journal of Clinical Oncology",
    # Radiology
    "Radiology",
    "Radiology: Artificial Intelligence",
]

# fetcher.execute(
#     venues=venues,
#     start_year=2022,
#     end_year=None,
#     query="",
# )

downloader.execute()
