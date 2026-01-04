"""
Configuration settings for the RAG Chatbot service
"""
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional



class Settings(BaseSettings):
    # OpenRouter Configuration
    openai_api_key: str | None = None
    openai_model: str = "llama-3.3-70b-versatile"
    openai_base_url: str = "https://api.groq.com/openai/v1"

    # Qdrant Configuration - remote/cloud from .env
    qdrant_url: str = Field("http://localhost:6333")
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "ai-book"

    # Application Configuration 
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False

    class Config:
        env_file = ".env"
        extra = "allow"

    @property
    def is_config_valid(self) -> bool:
        """Check if the basic configuration is valid"""
        return self.openai_api_key is not None and len(self.openai_api_key) > 0


settings = Settings()