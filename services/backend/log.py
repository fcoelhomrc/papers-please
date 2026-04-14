import logging

from pythonjsonlogger.json import JsonFormatter


def setup() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    # silence chatty third-party loggers
    for name in (
        "httpx",
        "httpcore",
        "sentence_transformers",
        "transformers",
        "docling",
        "pinecone",
        "urllib3",
        "uvicorn.access",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
