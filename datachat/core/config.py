# config.py
from dataclasses import dataclass
from typing import Optional
from functools import lru_cache
import os
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv


class Environment(str, Enum):
    """Application environment types"""

    PRODUCTION = "production"
    TEST = "test"


@dataclass(frozen=True)
class OpenAIConfig:
    """Configuration for OpenAI services"""

    api_key: str


@dataclass(frozen=True)
class PineconeConfig:
    """Configuration for Pinecone vector store"""

    api_key: str
    region: str


@dataclass(frozen=True)
class Config:
    """Application configuration"""

    openai: OpenAIConfig
    pinecone: PineconeConfig
    env: Environment

    @classmethod
    def load(cls, env: Environment = Environment.PRODUCTION) -> "Config":
        """Load configuration for specified environment"""
        return cls._load_config(env)

    @classmethod
    @lru_cache()
    def _load_config(cls, env: Environment) -> "Config":
        """Internal method to load configuration with caching"""
        # Determine config file path based on environment
        if env == Environment.TEST:
            env_path = Path(__file__).parent.parent.parent / "tests" / ".env.test"
        else:
            env_path = Path(__file__).parent.parent.parent / ".env"

        if not env_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {env_path}")

        # Load environment variables
        load_dotenv(env_path)

        # Verify required variables
        required_vars = {
            "OPENAI_API_KEY",
            "PINECONE_API_KEY",
            "PINECONE_REGION",
        }

        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"Please check your {env_path.name} file."
            )

        return cls(
            openai=OpenAIConfig(
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            pinecone=PineconeConfig(
                api_key=os.getenv("PINECONE_API_KEY"),
                region=os.getenv("PINECONE_REGION"),
            ),
            env=env,
        )
