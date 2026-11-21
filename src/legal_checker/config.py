"""Application settings loaded from environment variables (and an optional .env file)."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Runtime configuration.

    Provider API keys (e.g. ``ANTHROPIC_API_KEY``) and LangSmith variables are not
    stored here: LangChain reads them from the environment on its own.
    """

    model_config = ConfigDict(frozen=True)

    llm_model: str = Field("anthropic:claude-sonnet-5-5", description="init_chat_model 'provider:model' string")
    eli_api_base_url: str = "https://api.sejm.gov.pl/eli"
    bm25_top_k: int = Field(12, ge=1)
    agent_max_tool_calls: int = Field(4, ge=0)
    graph_recursion_limit: int = Field(20, ge=1)
    cache_dir: Path = Path(".cache")

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from the environment, loading ``.env`` first if present."""
        load_dotenv()
        overrides = {
            name: os.environ[name.upper()]
            for name in cls.model_fields
            if name.upper() in os.environ
        }
        return cls(**overrides)


def get_settings() -> Settings:
    return Settings.from_env()
