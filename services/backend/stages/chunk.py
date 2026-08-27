import logging
import time

import log
from config import load
from process.chunker import PdfChunker

log.setup()
logger = logging.getLogger("stage.chunk")


def run():
    cfg = load()
    s = cfg.stages.chunk
    with log.timed("chunk", limit=s.limit):
        PdfChunker().execute(limit=s.limit)


if __name__ == "__main__":
    cfg = load()
    interval_s = cfg.stages.chunk.interval_s
    logger.info("stage.chunk.loop_start", extra={"interval_s": interval_s})
    while True:
        try:
            run()
        except Exception:
            logger.exception("stage.chunk.error")
        time.sleep(interval_s)
