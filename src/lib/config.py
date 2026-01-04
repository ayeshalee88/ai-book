"""Configuration management for Qdrant retrieval validation tool."""

import os
from typing import Optional
from pydantic import BaseModel, Field


class QdrantConfig(BaseModel):
    """Configuration for Qdrant connection."""

    host: str = Field(default="localhost", description="Qdrant host address")
    port: int = Field(default=6333, description="Qdrant port number")
    api_key: Optional[str] = Field(default=None, description="Qdrant API key for authentication")
    collection_name: str = Field(default="documents", description="Name of the Qdrant collection")
    timeout: int = Field(default=30, description="Request timeout in seconds")

    @classmethod
    def from_env(cls) -> "QdrantConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION_NAME", "documents"),
            timeout=int(os.getenv("QDRANT_TIMEOUT", "30"))
        )