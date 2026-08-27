import logging
import time

import log
from config import load
from process.embedder import PdfEmbedder

log.setup()
logger = logging.getLogger("stage.embed")


def run():
    cfg = load()
    s = cfg.stages.embed
    with log.timed("embed", limit=s.limit):
        PdfEmbedder().execute(max_chunks=s.limit)


if __name__ == "__main__":
    cfg = load()
    interval_s = cfg.stages.embed.interval_s
    logger.info("stage.embed.loop_start", extra={"interval_s": interval_s})
    while True:
        try:
            run()
        except Exception:
            logger.exception("stage.embed.error")
        time.sleep(interval_s)
