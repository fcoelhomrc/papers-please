import logging
import time

import log
from config import load
from ingest.fetcher import PdfFetcher
from process.chunker import PdfChunker
from process.embedder import PdfEmbedder

log.setup()
logger = logging.getLogger("worker")


def run():
    cfg = load()
    w = cfg.worker

    logger.info("worker.start")

    with log.timed("download", limit=w.download_limit):
        PdfFetcher(max_workers=w.download_workers).execute(limit=w.download_limit)

    with log.timed("chunk", limit=w.chunk_limit):
        PdfChunker().execute(limit=w.chunk_limit)

    with log.timed("embed", limit=w.embed_limit):
        PdfEmbedder().execute(max_chunks=w.embed_limit)

    logger.info("worker.done")


if __name__ == "__main__":
    cfg = load()
    logger.info("worker.loop_start", extra={"interval_s": cfg.worker.interval_s})
    while True:
        try:
            run()
        except Exception:
            logger.exception("worker.error")
        time.sleep(cfg.worker.interval_s)
