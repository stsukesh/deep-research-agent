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