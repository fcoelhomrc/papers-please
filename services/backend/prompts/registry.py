"""Versioned prompt storage.

Every prompt sent to an LLM lives here as `<name>/<version>.md`, and edits
create a new version file rather than changing an existing one. The point is
reproducibility of eval scores: a report in `eval/reports/` records the
version it ran with, so a score stays attributable to an exact prompt, and
an old baseline stays comparable after the prompt is tuned.

Not stored here: `MODELS[...]["query_prompt"]` in process/embedder.py. That
is the prefix BGE's model card prescribes for query embeddings - it is bound
to the embedding model, not a prompt we're free to tune, and filing it
alongside these would imply otherwise.
"""
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str, version: str) -> str:
    path = PROMPTS_DIR / name / f"{version}.md"
    if not path.is_file():
        raise FileNotFoundError(
            f"no prompt {name!r} version {version!r} at {path} "
            f"(available: {', '.join(available_versions(name)) or 'none'})"
        )
    return path.read_text().strip()


def available_versions(name: str) -> list[str]:
    d = PROMPTS_DIR / name
    if not d.is_dir():
        return []
    # v2 before v10 - plain sort puts 'v10' first, which would make the
    # "latest version" read off this list wrong.
    return sorted((p.stem for p in d.glob("v*.md")), key=lambda v: int(v.lstrip("v")))
