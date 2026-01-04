"""Retrieved chunk model for Qdrant retrieval validation."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class RetrievedChunk(BaseModel):
    """Represents a chunk retrieved from Qdrant with its metadata."""

    id: str = Field(..., description="Unique identifier for the retrieved chunk")
    text: str = Field(..., description="The text content of the retrieved chunk")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score of the retrieval")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the chunk")
    position: int = Field(..., ge=0, description="Position in the retrieval results")
    collection_name: Optional[str] = Field(default=None, description="Name of the collection where chunk was found")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the chunk was retrieved")

    class Config:
        """Pydantic configuration."""
        extra = "forbid"