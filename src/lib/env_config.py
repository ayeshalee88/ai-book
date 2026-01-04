"""Environment configuration management for Qdrant validation tool."""

import os
from typing import Optional


class EnvironmentConfig:
    """Environment configuration management."""

    @staticmethod
    def get_qdrant_host() -> str:
        """Get Qdrant host from environment or default."""
        return os.getenv("QDRANT_HOST", "localhost")

    @staticmethod
    def get_qdrant_port() -> int:
        """Get Qdrant port from environment or default."""
        return int(os.getenv("QDRANT_PORT", "6333"))

    @staticmethod
    def get_qdrant_api_key() -> Optional[str]:
        """Get Qdrant API key from environment."""
        return os.getenv("QDRANT_API_KEY")

    @staticmethod
    def get_collection_name() -> str:
        """Get collection name from environment or default."""
        return os.getenv("QDRANT_COLLECTION_NAME", "documents")

    @staticmethod
    def get_timeout() -> int:
        """Get timeout from environment or default."""
        return int(os.getenv("QDRANT_TIMEOUT", "30"))

    @staticmethod
    def get_log_level() -> str:
        """Get log level from environment or default."""
        return os.getenv("LOG_LEVEL", "INFO")