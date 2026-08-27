import logging
import time

import log
from config import load
from ingest.fetcher import PdfFetcher

log.setup()
logger = logging.getLogger("stage.download")


def run():
    cfg = load()
    w = cfg.worker
    with log.timed("download", limit=w.download_limit):
        PdfFetcher(max_workers=w.download_workers).execute(limit=w.download_limit)


if __name__ == "__main__":
    cfg = load()
    logger.info("stage.download.loop_start", extra={"interval_s": cfg.worker.interval_s})
    while True:
        try:
            run()
        except Exception:
            logger.exception("stage.download.error")
        time.sleep(cfg.worker.interval_s)
