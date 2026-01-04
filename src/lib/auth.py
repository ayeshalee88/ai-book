"""Authentication utilities for Qdrant validation tool."""

import os
from typing import Optional
from qdrant_client.http.exceptions import UnexpectedResponse


def get_qdrant_api_key() -> Optional[str]:
    """Get Qdrant API key from environment variables."""
    return os.getenv("QDRANT_API_KEY")


def validate_api_key(api_key: Optional[str]) -> bool:
    """Validate that the API key is properly formatted."""
    if not api_key:
        return False

    # Basic validation: API key should be a non-empty string
    return isinstance(api_key, str) and len(api_key.strip()) > 0


def authenticate_qdrant_client(host: str, port: int, api_key: Optional[str] = None):
    """Authenticate with Qdrant and return a client instance."""
    from qdrant_client import QdrantClient

    client = QdrantClient(
        host=host,
        port=port,
        api_key=api_key
    )

    # Try a simple operation to verify authentication
    try:
        client.get_collections()
        return client
    except UnexpectedResponse as e:
        if e.status_code == 401:
            raise Exception("Authentication failed: Invalid API key")
        else:
            raise e
    except Exception as e:
        raise e


class AuthManager:
    """Manager for handling authentication-related operations."""

    @staticmethod
    def load_credentials_from_env() -> dict:
        """Load authentication credentials from environment variables."""
        return {
            "api_key": get_qdrant_api_key()
        }

    @staticmethod
    def validate_credentials(credentials: dict) -> bool:
        """Validate authentication credentials."""
        api_key = credentials.get("api_key")
        return validate_api_key(api_key)