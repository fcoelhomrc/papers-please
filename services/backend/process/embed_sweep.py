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

# Below this a batch is not worth keeping on the card: the per-batch Postgres
# and Pinecone round trips start to dominate, and a model that cannot fit two
# chunks at a time is telling us something a smaller batch will not fix.
MIN_BATCH = 2


def _release_card() -> None:
    """Hand the card back before the next encoder allocates on it.

    Torch keeps freed blocks in its caching allocator, so dropping the last
    reference to a model is not by itself enough to make room for the next
    one.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def embed(model_key: str, device: str, batch_size: int) -> None:
    """Embed every chunk still pending under `model_key`."""
    from config import load

    load().devices.embedder = device
    logger.info(
        "embed_sweep.start",
        extra={"model": model_key, "device": device, "batch_size": batch_size},
    )
    embedder = PdfEmbedder(model_key=model_key, batch_size=batch_size)
    try:
        embedder.execute(max_chunks=ALL_CHUNKS)
    finally:
        del embedder
        _release_card()


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def embed_with_fallback(model_key: str, device: str, fallback: bool) -> str:
    """Embed under `model_key`, giving up as little as possible on each OOM.

    Halve the batch before surrendering the card. Measured on this corpus,
    qwen3-0.6b ran ~1,200 chunks in four minutes on the GPU and 32 chunks in
    four minutes on the CPU — a 21-hour job against a 30-minute one — so a
    fallback that goes straight to the CPU turns a batch size that is slightly
    too big into an overnight run that does not finish.

    Returns what actually finished the work: the sweep is only comparable
    across models if that is written down.
    """
    batch_size = MODELS[model_key]["batch_size"]
    while True:
        try:
            embed(model_key, device, batch_size)
            return f"{device} (batch {batch_size})"
        except Exception as exc:
            if not (fallback and _is_oom(exc)):
                raise
            # Chunks embedded before the OOM are already recorded in
            # chunk_embeddings, so each retry resumes rather than starting over.
            _release_card()
            if device != "cpu" and batch_size // 2 >= MIN_BATCH:
                batch_size //= 2
                logger.warning(
                    "embed_sweep.oom_smaller_batch",
                    extra={"model": model_key, "batch_size": batch_size},
                )
                continue
            if device != "cpu":
                logger.warning("embed_sweep.oom_to_cpu", extra={"model": model_key})
                device, batch_size = "cpu", MODELS[model_key]["batch_size"]
                continue
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Embed the corpus under several models, in sequence"
    )
    parser.add_argument("models", nargs="+", choices=sorted(MODELS), metavar="MODEL")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="fail instead of retrying an OOM on a smaller batch or the CPU",
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
