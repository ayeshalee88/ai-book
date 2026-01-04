"""Query model for Qdrant retrieval validation."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class Query(BaseModel):
    """Represents a query for Qdrant retrieval validation."""

    text: str = Field(..., description="The query text for semantic search")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to retrieve")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters for the query")
    query_vector: Optional[List[float]] = Field(default=None, description="Optional pre-computed query vector")

    class Config:
        """Pydantic configuration."""
        extra = "forbid"