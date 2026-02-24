"""Application configuration."""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class Config:
    """Central config for the RAG application."""

    # API keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # LLM
    LLM_MODEL: str = "gpt-4o-mini"

    # Document processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
    ]

    @classmethod
    def get_llm(cls, model: str | None = None, temperature: float = 0):
        """Initialise and return the LLM model."""
        return ChatOpenAI(
            model=model or cls.LLM_MODEL,
            temperature=temperature,
        )