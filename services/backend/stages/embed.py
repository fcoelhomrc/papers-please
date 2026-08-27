import logging
import time

import log
from config import load
from process.embedder import PdfEmbedder

log.setup()
logger = logging.getLogger("stage.embed")


def run():
    cfg = load()
    w = cfg.worker
    with log.timed("embed", limit=w.embed_limit):
        PdfEmbedder().execute(max_chunks=w.embed_limit)


if __name__ == "__main__":
    cfg = load()
    logger.info("stage.embed.loop_start", extra={"interval_s": cfg.worker.interval_s})
    while True:
        try:
            run()
        except Exception:
            logger.exception("stage.embed.error")
        time.sleep(cfg.worker.interval_s)
