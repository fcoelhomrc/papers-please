import os

import yaml
from pydantic import BaseModel


class DatabaseConfig(BaseModel):
    host: str = "localhost"


class StorageConfig(BaseModel):
    root: str = "data"


class EmbedderConfig(BaseModel):
    model: str = "bge-small"
    max_chunks: int = 1_000


class SearchConfig(BaseModel):
    top_k: int = 10
    rerank_top_k: int = 5


class Config(BaseModel):
    database: DatabaseConfig = DatabaseConfig()
    storage: StorageConfig = StorageConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    search: SearchConfig = SearchConfig()


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
