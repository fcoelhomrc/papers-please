import logging
import time
from contextlib import contextmanager
from typing import Generator

from pythonjsonlogger.json import JsonFormatter

_logger = logging.getLogger("worker")


def setup() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

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


@contextmanager
def timed(step: str, **extra) -> Generator[None, None, None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        _logger.info(step, extra={"duration_s": round(time.perf_counter() - start, 2), **extra})
