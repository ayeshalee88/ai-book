"""Validation log model for Qdrant retrieval validation."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from .retrieved_chunk import RetrievedChunk
from .query import Query


class ValidationLog(BaseModel):
    """Represents a log entry for a Qdrant retrieval validation operation."""

    id: str = Field(..., description="Unique identifier for the validation log")
    query: Query = Field(..., description="The query that was executed")
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list, description="Chunks retrieved from Qdrant")
    execution_time_ms: float = Field(..., ge=0, description="Time taken to execute the query in milliseconds")
    success: bool = Field(..., description="Whether the validation was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if validation failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the validation was executed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the validation")

    class Config:
        """Pydantic configuration."""
        extra = "forbid"