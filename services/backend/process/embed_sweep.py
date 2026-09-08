"""Embed the corpus under several models, one after another.

The card fits exactly one encoder at a time — 4GB total, and the models being
compared cost 0.5–2.5GB each — so the sweep runs them in sequence. Two workers
in parallel would OOM the second one and leave a half-filled index, which shows
up later as a bad model rather than as a scheduling mistake.

The device is named here rather than read from `config.yaml`. Embedding a
corpus is a batch job that wants the GPU; answering queries during an eval
wants the CPU, because the backend container has no card and every latency
figure so far was measured with the query encoder on CPU. Those are two
different jobs with two different answers, so the sweep carries its own
instead of making the config field mean whichever one ran last.

    python -m process.embed_sweep arctic-m-v2 qwen3-0.6b
"""
import argparse
import gc
import logging

import log
import torch
from process.embedder import MODELS, PdfEmbedder

log.setup()
logger = logging.getLogger("stage.embed_sweep")

# Everything pending in one pass. `stages.embed.limit` exists to keep a polling
# loop responsive between naps; a sweep has nothing to stay responsive for.
ALL_CHUNKS = 1_000_000


def _release_card() -> None:
    """Hand the card back before the next encoder allocates on it.

    Torch keeps freed blocks in its caching allocator, so dropping the last
    reference to a model is not by itself enough to make room for the next
    one.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def embed(model_key: str, device: str) -> None:
    """Embed every chunk still pending under `model_key`."""
    from config import load

    load().devices.embedder = device
    logger.info("embed_sweep.start", extra={"model": model_key, "device": device})
    embedder = PdfEmbedder(model_key=model_key)
    try:
        embedder.execute(max_chunks=ALL_CHUNKS)
    finally:
        del embedder
        _release_card()


def embed_with_fallback(model_key: str, device: str, fallback: bool) -> str:
    """Embed under `model_key`, retrying on the CPU if the card is too small.

    Returns the device that finished the work — the sweep is only comparable
    across models if it is written down, and qwen3-0.6b is expected to be the
    one that needs it.
    """
    try:
        embed(model_key, device)
        return device
    except Exception as exc:
        if not (fallback and device != "cpu" and "out of memory" in str(exc).lower()):
            raise
        # Chunks embedded before the OOM are already recorded in
        # chunk_embeddings, so this resumes rather than starting over.
        logger.warning("embed_sweep.oom_fallback", extra={"model": model_key})
        _release_card()
        embed(model_key, "cpu")
        return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Embed the corpus under several models, in sequence"
    )
    parser.add_argument("models", nargs="+", choices=sorted(MODELS), metavar="MODEL")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="fail instead of retrying an OOM on the CPU",
    )
    args = parser.parse_args()

    outcomes: dict[str, str] = {}
    for model_key in args.models:
        # One model failing does not cancel the rest: the sweep is meant to be
        # left alone overnight, and three encoders measured beats none.
        try:
            with log.timed("embed_sweep.model", model=model_key):
                outcomes[model_key] = embed_with_fallback(
                    model_key, args.device, not args.no_cpu_fallback
                )
        except Exception:
            logger.exception("embed_sweep.failed", extra={"model": model_key})
            outcomes[model_key] = "failed"

    logger.info("embed_sweep.done", extra={"outcomes": outcomes})
    if any(v == "failed" for v in outcomes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
