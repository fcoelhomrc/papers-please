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
    max_tokens: int = 512
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


class StagesConfig(BaseModel):
    download: DownloadStageConfig = DownloadStageConfig(limit=20)
    chunk: StageConfig = StageConfig(limit=10)
    embed: StageConfig = StageConfig(limit=500)


class LLMConfig(BaseModel):
    provider: str = "anthropic"  # "anthropic" | "vllm"
    model: str = "claude-haiku-4-5"
    max_tokens: int = 512  # keep replies (and cost) bounded - this agent's replies are short
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
