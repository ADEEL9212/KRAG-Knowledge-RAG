"""
Configuration management using Pydantic settings.
"""

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")

    # Vector Store Configuration
    vector_store_type: str = Field(
        default="chroma", description="Vector store type (chroma or pinecone)"
    )
    chroma_persist_directory: str = Field(
        default="./data/chroma", description="ChromaDB persistence directory"
    )
    pinecone_api_key: str = Field(default="", description="Pinecone API key")
    pinecone_environment: str = Field(default="", description="Pinecone environment")
    pinecone_index_name: str = Field(
        default="knowledge-rag", description="Pinecone index name"
    )

    # Embedding Model Configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence Transformer model name"
    )
    embedding_dimension: int = Field(
        default=384, description="Embedding vector dimension"
    )

    # Chunking Configuration
    chunk_size: int = Field(default=500, description="Text chunk size in characters")
    chunk_overlap: int = Field(
        default=50, description="Overlap between chunks in characters"
    )

    # Retrieval Configuration
    top_k: int = Field(default=5, description="Number of results to retrieve")
    similarity_threshold: float = Field(
        default=0.7, description="Minimum similarity score threshold"
    )

    # LLM Configuration
    llm_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model name")
    llm_temperature: float = Field(
        default=0.7, description="LLM temperature for generation"
    )
    max_tokens: int = Field(
        default=500, description="Maximum tokens for LLM response"
    )

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_reload: bool = Field(default=True, description="Enable auto-reload")
    cors_origins: List[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="text", description="Log format (json or text)")


# Global settings instance
settings = Settings()
