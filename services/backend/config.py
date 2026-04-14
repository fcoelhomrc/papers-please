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


class WorkerConfig(BaseModel):
    interval_s: int = 300
    download_workers: int = 4
    download_limit: int = 20
    chunk_limit: int = 10
    embed_limit: int = 500


class Config(BaseModel):
    database: DatabaseConfig = DatabaseConfig()
    storage: StorageConfig = StorageConfig()
    devices: DevicesConfig = DevicesConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    search: SearchConfig = SearchConfig()
    worker: WorkerConfig = WorkerConfig()


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
