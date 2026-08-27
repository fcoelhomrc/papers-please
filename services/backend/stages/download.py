import logging
import time

import log
from config import load
from ingest.fetcher import PdfFetcher

log.setup()
logger = logging.getLogger("stage.download")


def run():
    cfg = load()
    s = cfg.stages.download
    with log.timed("download", limit=s.limit):
        PdfFetcher(max_workers=s.workers).execute(limit=s.limit)


if __name__ == "__main__":
    cfg = load()
    interval_s = cfg.stages.download.interval_s
    logger.info("stage.download.loop_start", extra={"interval_s": interval_s})
    while True:
        try:
            run()
        except Exception:
            logger.exception("stage.download.error")
        time.sleep(interval_s)
