"""Where a run's output goes: one directory per embedding model.

Every number this project reports is only comparable to others measured on the
same embedding model - the chunk ids are stable across a re-embed, but the
vectors, and therefore the ranking, are not. A flat results directory made
that invisible: two `ablation-all-*.json` files sorted by timestamp, one from
bge-small and one from something else, and the newest won.

So the model is a directory rather than a field to remember to check. Sweeping
embedders adds directories instead of overwriting the previous model's
evidence, which is what makes a cross-model comparison possible at the end
rather than a sequence of runs that each replaced the last.

The database and Pinecone already work this way - chunk_embeddings is keyed
(chunk_id, model_id) and each model has its own index - so this is the results
layer catching up with the storage layer.
"""
from pathlib import Path

RESULTS_ROOT = Path(__file__).parent / "results"


def results_dir(embed_model: str | None = None) -> Path:
    """The directory for `embed_model`, defaulting to the configured one."""
    if embed_model is None:
        from config import load

        embed_model = load().embedder.model
    path = RESULTS_ROOT / embed_model
    path.mkdir(parents=True, exist_ok=True)
    return path


def known_models() -> list[str]:
    """Embedding models that have results on disk."""
    return sorted(p.name for p in RESULTS_ROOT.iterdir()
                  if p.is_dir() and any(p.glob("*.json")))
