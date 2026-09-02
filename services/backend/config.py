import os

import yaml
from pydantic import BaseModel


class DatabaseConfig(BaseModel):
    host: str = "db"


class StorageConfig(BaseModel):
    root: str = "data"


class DevicesConfig(BaseModel):
    chunker: str = "cpu"
    embedder: str = "cpu"
    reranker: str = "cuda"


class EmbedderConfig(BaseModel):
    model: str = "bge-small"
    # Chunk size, in the embedding model's own tokens. 256 rather than
    # bge-small's full 512 window: a smaller chunk is a sharper vector,
    # because averaging a passage that covers two topics lands the embedding
    # between both and near neither. The context a generator needs is
    # recovered after ranking instead, by search.neighbour_window - which is
    # what makes shrinking this affordable rather than lossy.
    max_tokens: int = 256
    max_chunks: int = 1_000


class SearchConfig(BaseModel):
    top_k: int = 10
    rerank_top_k: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # "semantic" | "keyword" | "hybrid". Hybrid, because eval says it wins:
    # at top_k=5 it scores recall 0.833 / nDCG 0.814 / MRR 0.820 against
    # semantic-only's 0.810 / 0.787 / 0.788 (eval/reports/, #23).
    mode: str = "hybrid"
    rrf_k: int = 60  # RRF damping; 60 is the original paper's default
    hybrid_candidates: int = 20  # per-source pool size before fusion narrows to top_k
    # How many candidates to retrieve before the cross-encoder cuts down to
    # the requested top_k, for callers that ask for "the best k" rather than
    # naming exact retrieval sizes (search_chunks, FixedPipeline).
    #
    # Reranking only pays off when it gets more candidates than it returns.
    # Retrieving 5 and reranking to 5 reorders those 5 and can never rescue a
    # 6th - which is what shipped, so the cross-encoder was doing a fraction
    # of the work it was loaded for. The measured headroom is in the ledger:
    # hybrid top_k=50 unranked recalls 0.940 against top_k=10 rerank->10's
    # 0.893, so the relevant chunk is usually *in* a wide pool and merely
    # ranked too low to survive a narrow one.
    #
    # 40 rather than 50: the cross-encoder is O(candidates) local CPU on
    # every search, and recall gains flatten well before 50 while latency
    # does not.
    rerank_candidates: int = 40
    # How many chunks either side of a hit to glue on as context (0 = off).
    #
    # HybridChunker emits no overlap, so a sentence spanning a boundary is
    # split and neither half reads as an answer on its own. Expanding after
    # ranking - never before - keeps the cross-encoder scoring the precise
    # chunk while generation gets the surrounding prose.
    #
    # Paired with the drop in embedder.max_tokens (512 -> 256): a 3-chunk
    # window of 256-token chunks is about the same context as one old
    # 512-token chunk, so this buys retrieval precision rather than spending
    # tokens. Widening the window without shrinking the chunks would just be
    # a bigger bill.
    neighbour_window: int = 1
    # Keyword's weight in RRF, relative to dense retrieval's 1.0. Not 1.0
    # because the two are not equally good rankers (nDCG 0.598 vs 0.787 alone
    # at top_k=5), and weighting them equally measured *worse* than dense
    # alone. Chosen from a sweep on the 50-question eval set, where the trend
    # from 1.0 down to 0.1 is monotonic - a small set, so treat it as a
    # sensible default rather than a tuned constant.
    keyword_weight: float = 0.1
    # Minimum scores, each in its own source's units - cosine, ts_rank and
    # cross-encoder logits are not on comparable scales. None = no floor,
    # which is what shipped: retrieval always returned top_k and so could
    # never answer "nothing here is relevant".
    min_vector_score: float | None = None
    min_keyword_score: float | None = None
    # -8.0 is the knee of the measured curve: abstention_precision goes
    # 0.000 -> 0.500 and mean false positives 3.12 -> 0.62 at *zero* recall
    # cost (0.821 either way). Tightening to -5.0 buys abstention 0.875 but
    # costs ~7 points of recall, so it's left as a deliberate choice.
    #
    # This number is in the cross-encoder's own logit units and is specific to
    # ms-marco-MiniLM-L-6-v2 - changing search.reranker_model invalidates it
    # and it must be re-swept (eval/thresholds.py). Only applies when rerank
    # is on; there is no equivalent floor on the bi-encoder, whose scores for
    # relevant and irrelevant queries overlap heavily.
    min_rerank_score: float | None = -8.0


class StageConfig(BaseModel):
    interval_s: int = 300
    limit: int = 20


class DownloadStageConfig(StageConfig):
    workers: int = 4
    # How many times to retry a PDF URL before giving up on it. Some
    # open-access links are permanently broken - a DOI resolver that 403s to
    # a landing page will never succeed - and without a cap the stage
    # re-attempted them on every pass forever, hammering the remote host.
    # Three, matching the chunk stage: the failures worth retrying are
    # transient (a flaky host, a timeout) and clear well inside three passes.
    max_attempts: int = 3


class ChunkStageConfig(StageConfig):
    # How many times to retry a PDF that failed to chunk before giving up on
    # it. OCR is the most expensive step in the pipeline, so a file that will
    # never parse must not be allowed to consume it indefinitely. Three
    # because the failures worth retrying are transient (a truncated
    # download, a memory spike) and those clear well inside three passes.
    max_attempts: int = 3


class StagesConfig(BaseModel):
    download: DownloadStageConfig = DownloadStageConfig(limit=20)
    chunk: ChunkStageConfig = ChunkStageConfig(limit=10)
    embed: StageConfig = StageConfig(limit=500)


class LLMConfig(BaseModel):
    """Which model answers, and through whom.

    OpenRouter is the default because it turns "which model" into a config
    string instead of a code change: one OpenAI-compatible endpoint fronting
    every provider, so switching from a free model to Claude is editing
    `model` rather than editing llm.py. The anthropic and vllm branches are
    kept and still work - they are just not the default any more.
    """

    provider: str = "openrouter"  # "openrouter" | "anthropic" | "vllm"

    # OpenRouter model ids are namespaced `vendor/model`, and a `:free`
    # suffix selects the free tier of that model. Free models are rate
    # limited hard enough to matter - see the README - so this is a "does it
    # work" default, not a "run the whole eval" one.
    model: str = "minimax/minimax-m2.7:free"

    # The eval judge, kept separate from the pipeline model on purpose.
    # Changing the model that *answers* is an experiment; changing the model
    # that *scores* silently re-baselines every historical number in
    # eval/ledger.jsonl, so the two must be movable independently. Empty
    # means "same as model".
    #
    # deepseek-v4-flash was measured against the incumbent claude-haiku-4.5
    # on three known-answer faithfulness cases and returned identical scores
    # at 1/15th the cost (docs/judge-model-selection.md). Deliberately NOT
    # the pipeline model: a model grading its own output is self-preference
    # bias, and it would inflate the exact metric this project has a known
    # problem with.
    judge_model: str = "deepseek/deepseek-v4-flash"

    max_tokens: int = 512  # keep replies (and cost) bounded - this agent's replies are short

    openrouter_url: str = "https://openrouter.ai/api/v1"
    vllm_url: str = "http://localhost:8001/v1"
    vllm_model: str = ""


class PromptsConfig(BaseModel):
    """Which version of each prompt to load from prompts/<name>/<version>.md.
    Config rather than a constant so an eval run can score a candidate prompt
    (`--prompt-version orchestrator=v2`) without editing code."""

    orchestrator: str = "v1"
    fixed_rag: str = "v1"


class Config(BaseModel):
    database: DatabaseConfig = DatabaseConfig()
    storage: StorageConfig = StorageConfig()
    devices: DevicesConfig = DevicesConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    search: SearchConfig = SearchConfig()
    stages: StagesConfig = StagesConfig()
    llm: LLMConfig = LLMConfig()
    prompts: PromptsConfig = PromptsConfig()


_config: Config | None = None


# singleton pattern
def load(path: str | None = None) -> Config:
    global _config
    if _config is not None:
        return _config
    path = path or os.environ.get("CONFIG_PATH", "config.yaml")
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f)
        _config = Config.model_validate(data or {})
    else:
        _config = Config()
    return _config
