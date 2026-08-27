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


class Config(BaseModel):
    database: DatabaseConfig = DatabaseConfig()
    storage: StorageConfig = StorageConfig()
    devices: DevicesConfig = DevicesConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    search: SearchConfig = SearchConfig()
    stages: StagesConfig = StagesConfig()
    llm: LLMConfig = LLMConfig()


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
