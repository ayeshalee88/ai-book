"""Performance metrics model for Qdrant retrieval validation."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class PerformanceMetrics(BaseModel):
    """Represents performance metrics for Qdrant retrieval validation."""

    id: str = Field(..., description="Unique identifier for the metrics record")
    query_latency_ms: float = Field(..., ge=0, description="Query execution latency in milliseconds")
    p95_latency_ms: Optional[float] = Field(default=None, description="95th percentile latency")
    p99_latency_ms: Optional[float] = Field(default=None, description="99th percentile latency")
    throughput_qps: Optional[float] = Field(default=None, description="Queries per second throughput")
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Success rate of queries")
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Error rate of queries")
    total_queries: int = Field(default=0, ge=0, description="Total number of queries executed")
    cache_hit_rate: Optional[float] = Field(default=None, description="Cache hit rate if applicable")
    memory_usage_mb: Optional[float] = Field(default=None, description="Memory usage in MB")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When metrics were recorded")
    additional_metrics: Dict[str, Any] = Field(default_factory=dict, description="Additional custom metrics")

    class Config:
        """Pydantic configuration."""
        extra = "forbid"