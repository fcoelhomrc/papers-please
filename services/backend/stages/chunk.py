import logging
import time

import log
from config import load
from process.chunker import PdfChunker

log.setup()
logger = logging.getLogger("stage.chunk")


def run():
    cfg = load()
    w = cfg.worker
    with log.timed("chunk", limit=w.chunk_limit):
        PdfChunker().execute(limit=w.chunk_limit)


if __name__ == "__main__":
    cfg = load()
    logger.info("stage.chunk.loop_start", extra={"interval_s": cfg.worker.interval_s})
    while True:
        try:
            run()
        except Exception:
            logger.exception("stage.chunk.error")
        time.sleep(cfg.worker.interval_s)
