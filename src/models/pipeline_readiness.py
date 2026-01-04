"""Pipeline readiness model for Qdrant retrieval validation."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from src.models.validation_log import ValidationLog
from src.models.performance_metrics import PerformanceMetrics


class ValidationCheck(BaseModel):
    """Represents a single validation check in the pipeline."""

    name: str = Field(..., description="Name of the validation check")
    description: str = Field(..., description="Description of what the check validates")
    success: bool = Field(..., description="Whether the check passed")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details about the check")
    execution_time_ms: Optional[float] = Field(default=None, description="Time taken to execute the check in milliseconds")


class PipelineReadiness(BaseModel):
    """Represents the readiness status of the entire Qdrant retrieval pipeline."""

    id: str = Field(..., description="Unique identifier for the readiness report")
    overall_status: str = Field(..., description="Overall readiness status (ready, not_ready, partial)")
    checks: List[ValidationCheck] = Field(default_factory=list, description="List of validation checks performed")
    validation_logs: List[ValidationLog] = Field(default_factory=list, description="Validation logs from the checks")
    performance_metrics: Optional[PerformanceMetrics] = Field(default=None, description="Overall performance metrics")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the readiness check was performed")
    summary: Optional[str] = Field(default=None, description="Summary of the readiness status")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improving readiness")

    class Config:
        """Pydantic configuration."""
        extra = "forbid"

    @property
    def passed_checks_count(self) -> int:
        """Count of checks that passed."""
        return sum(1 for check in self.checks if check.success)

    @property
    def failed_checks_count(self) -> int:
        """Count of checks that failed."""
        return sum(1 for check in self.checks if not check.success)

    @property
    def success_rate(self) -> float:
        """Success rate of all checks."""
        if not self.checks:
            return 0.0
        return self.passed_checks_count / len(self.checks)