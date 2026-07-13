"""Central settings + DB helper.

Two-layer config:
  config.yaml  — structure, defaults, and env-var *names* (commit this)
  .env          — actual values: LLM_MODEL, LLM_BASE_URL, etc. (never commit)

One change in .env propagates to every LLM call in the project.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# config.yaml loader (cached; re-read only if the file changes)
# ---------------------------------------------------------------------------

_YAML_PATH = Path(__file__).parent / "config.yaml"


@lru_cache(maxsize=1)
def _load_yaml() -> dict[str, Any]:
    try:
        import yaml  # PyYAML
        with open(_YAML_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def get_llm_config() -> dict[str, Any]:
    """Return resolved LLM config: values read from env via names declared in config.yaml."""
    cfg = _load_yaml().get("llm", {})

    provider  = os.environ.get("LLM_PROVIDER",  cfg.get("provider",  "local"))
    base_url  = os.environ.get(cfg.get("base_url_env", "LLM_BASE_URL"), "http://localhost:11434/v1")
    model     = os.environ.get(cfg.get("model_env",    "LLM_MODEL"),    "qwen3:8b")
    api_key   = os.environ.get(cfg.get("api_key_env",  "LLM_API_KEY"),  "local")

    # Vision model — falls back to the main LLM model if not set separately
    vis_env   = _load_yaml().get("vision", {}).get("model_env", "LLM_VISION_MODEL")
    vis_model = os.environ.get(vis_env, "") or model

    return {
        "provider":    provider,
        "base_url":    base_url,
        "model":       model,
        "api_key":     api_key,
        "vision_model": vis_model,
        "temperature": cfg.get("temperature", 0.1),
        "max_tokens":  cfg.get("max_tokens",  4096),
        "no_think":    cfg.get("no_think",    False),
    }


# ---------------------------------------------------------------------------
# Pydantic settings (still used for DB, embeddings, app settings)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/twostroke"

    # llm — resolved dynamically via get_llm_config(); kept here for code
    # that accesses settings.llm_model directly (populated from config.yaml + env)
    @property
    def llm_provider(self) -> str:   return get_llm_config()["provider"]
    @property
    def llm_base_url(self) -> str:   return get_llm_config()["base_url"]
    @property
    def llm_model(self) -> str:      return get_llm_config()["model"]
    @property
    def llm_api_key(self) -> str:    return get_llm_config()["api_key"]
    @property
    def llm_vision_model(self) -> str: return get_llm_config()["vision_model"]
    # kept for backward compat with any code that imports openai_api_key
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # embeddings
    @property
    def embedding_model(self) -> str:
        emb = _load_yaml().get("embeddings", {})
        return os.environ.get(emb.get("model_env", "EMBEDDING_MODEL"),
                              emb.get("default_model", "intfloat/multilingual-e5-small"))

    @property
    def embedding_dim(self) -> int:
        emb = _load_yaml().get("embeddings", {})
        raw = os.environ.get(emb.get("dim_env", "EMBEDDING_DIM"), "")
        return int(raw) if raw.isdigit() else int(emb.get("default_dim", 384))

    # reranker (not in yaml yet — keep as plain fields)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # latency controls
    bm25_cache_ttl_seconds: int = 300
    fast_path_enabled: bool = True
    verify_enabled: bool = True
    quality_boost: float = 0.15
    quality_penalty: float = 0.25
    downvote_cutoff: float = -0.5

    @property
    def retrieve_top_k(self) -> int:
        return int(_load_yaml().get("retrieval", {}).get("top_k", 20))

    @property
    def rerank_top_k(self) -> int:
        return int(_load_yaml().get("retrieval", {}).get("rerank_k", 5))

    # ingestion
    pdf_max_pages: int = 500

    @property
    def dedup_cosine_threshold(self) -> float:
        return float(_load_yaml().get("ingestion", {}).get("dedup_threshold", 0.98))

    @property
    def ocr_langs(self) -> str:
        return _load_yaml().get("ingestion", {}).get("ocr_langs", "deu+eng")

    @property
    def upload_dir(self) -> str:
        return os.environ.get("UPLOAD_DIR",
               _load_yaml().get("ingestion", {}).get("upload_dir", "./data/uploads"))

    image_store_path: str = "./data/images"
    enable_figure_handler: bool = False

    # app
    @property
    def app_host(self) -> str:
        return _load_yaml().get("app", {}).get("host", "0.0.0.0")

    @property
    def app_port(self) -> int:
        return int(_load_yaml().get("app", {}).get("port", 8000))


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_connection():
    """Return a psycopg connection. TODO: pool for production."""
    import psycopg  # local import so the module loads even before deps installed

    # Windows localhost resolution can try an unreachable address first, which
    # makes graph endpoints appear to hang even though Postgres is healthy.
    conninfo = get_settings().database_url.replace("@localhost:", "@127.0.0.1:")
    return psycopg.connect(conninfo, connect_timeout=5)
