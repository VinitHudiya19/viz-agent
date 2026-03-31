import logging
import sys
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


# ── Structured Logging ────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    """Configure structured logging for the Visualization Agent."""
    logger = logging.getLogger("viz-agent")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# ── Settings ──────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.local", env_file_encoding="utf-8", extra="ignore")

    LLM_PROVIDER: Literal["ollama", "openai", "anthropic", "groq", "grok", "azure_openai"] = "azure_openai"
    GROQ_API_KEY: str = ""
    XAI_API_KEY: str = ""
    GROQ_MODEL: str = ""
    OLLAMA_MODEL: str = ""
    OLLAMA_BASE_URL: str = ""
    OPENAI_API_KEY: str = ""
    # Azure OpenAI
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_DEPLOYMENT_NAME: str = ""
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    STORAGE_TYPE: Literal["local", "azure_blob"] = "local"
    AZURE_STORAGE_CONNECTION_STRING: str = ""
    AZURE_STORAGE_CONTAINER_NAME: str = "charts"
    CHART_OUTPUT_PATH: str = "./charts"
    MAX_DATA_ROWS: int = 1000
    PORT: int = 8003


settings = Settings()
logger = setup_logging()