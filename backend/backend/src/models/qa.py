"""
Data models for the QA API
"""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class QARequest(BaseModel):
    """
    Request model for the QA endpoint
    """
    question: str
    top_k: Optional[int] = 5  # Number of relevant chunks to retrieve


class SourceDetails(BaseModel):
    """
    Model for detailed source information
    """
    source: str
    section: str
    url: str
    title: str
    page: str
    full_content_length: int
    additional_metadata: dict = {}


class SourceInfo(BaseModel):
    """
    Model for source information
    """
    content_snippet: str
    score: float
    source_metadata: Dict[str, Any]
    source_details: Optional[SourceDetails] = None


class QAResponse(BaseModel):
    """
    Response model for the QA endpoint
    """
    question: str
    answer: str
    sources: List[SourceInfo]
    retrieved_chunks_count: int
    error: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """
    Response model for health check endpoint
    """
    status: str
    services: Dict[str, bool]