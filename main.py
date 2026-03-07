from services.data.fetcher import Fetcher

fetcher = Fetcher()

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

fetcher.fetch_batch(
    venues=venues,
    start_year=2022,
    end_year=None,
    query="",
)
