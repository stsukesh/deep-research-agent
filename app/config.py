"""
Configuration Management
========================
WHAT: Centralizes all app configuration (API keys, DB URLs, model names).
HOW:  Pydantic BaseSettings auto-reads from environment variables or .env files.
WHY:  12-factor app pattern — config lives in the environment, never hardcoded.
      This means the same code runs in dev, staging, and production with different
      env vars. Docker Compose injects these via the `environment:` section.

INTERVIEW Q&A:
  Q: How do you manage secrets and configuration in production?
  A: I use Pydantic BaseSettings which reads from environment variables.
     API keys are injected via Docker Compose env vars (or Kubernetes secrets).
     Never committed to git — only .env.example is tracked. The BaseSettings
     class provides type validation, defaults, and documentation all in one place.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ----- LLM -----
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    NVIDIA_API_KEY: str = ""
    NVIDIA_MODEL: str = "meta/llama-3.3-70b-instruct"

    # ----- Search -----
    TAVILY_API_KEY: str = ""

    # ----- Database -----
    DATABASE_URL: str = "postgresql+asyncpg://research:research@localhost:5432/research_db"
    CHECKPOINT_DB_URL: str = "postgresql://research:research@localhost:5432/research_db"

    # ----- App -----
    APP_ENV: str = "development"
    MAX_REVISIONS: int = 3

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # Ignore unknown env vars
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings singleton.
    lru_cache ensures we only parse env vars once, not on every request.
    """
    return Settings()
