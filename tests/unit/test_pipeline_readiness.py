"""Unit tests for PipelineReadiness model."""

import pytest
from datetime import datetime
from src.models.pipeline_readiness import PipelineReadiness, ValidationCheck
from src.models.validation_log import ValidationLog
from src.models.query import Query
from src.models.performance_metrics import PerformanceMetrics


class TestPipelineReadinessModel:
    """Test cases for the PipelineReadiness model."""

    def test_pipeline_readiness_creation_valid(self):
        """Test creating a valid PipelineReadiness instance."""
        check = ValidationCheck(
            name="connection_check",
            description="Check if Qdrant is accessible",
            success=True
        )

        readiness = PipelineReadiness(
            id="readiness_1",
            overall_status="ready",
            checks=[check]
        )

        assert readiness.id == "readiness_1"
        assert readiness.overall_status == "ready"
        assert len(readiness.checks) == 1
        assert readiness.checks[0].name == "connection_check"
        assert isinstance(readiness.timestamp, datetime)

    def test_validation_check_creation(self):
        """Test creating a ValidationCheck instance."""
        check = ValidationCheck(
            name="retrieval_test",
            description="Test if retrieval works correctly",
            success=True,
            details={"top_k": 5, "results": 3},
            execution_time_ms=100.5
        )

        assert check.name == "retrieval_test"
        assert check.success is True
        assert check.details == {"top_k": 5, "results": 3}
        assert check.execution_time_ms == 100.5

    def test_pipeline_readiness_with_validation_logs(self):
        """Test creating PipelineReadiness with validation logs."""
        check = ValidationCheck(
            name="basic_check",
            description="Basic functionality check",
            success=True
        )

        query = Query(text="test query", top_k=5)
        validation_log = ValidationLog(
            id="log_1",
            query=query,
            execution_time_ms=100.0,
            success=True
        )

        readiness = PipelineReadiness(
            id="readiness_1",
            overall_status="ready",
            checks=[check],
            validation_logs=[validation_log]
        )

        assert len(readiness.validation_logs) == 1
        assert readiness.validation_logs[0].id == "log_1"

    def test_pipeline_readiness_properties(self):
        """Test the calculated properties of PipelineReadiness."""
        checks = [
            ValidationCheck(name="check1", description="Check 1", success=True),
            ValidationCheck(name="check2", description="Check 2", success=False),
            ValidationCheck(name="check3", description="Check 3", success=True),
            ValidationCheck(name="check4", description="Check 4", success=False)
        ]

        readiness = PipelineReadiness(
            id="readiness_1",
            overall_status="partial",
            checks=checks
        )

        assert readiness.passed_checks_count == 2
        assert readiness.failed_checks_count == 2
        assert readiness.success_rate == 0.5  # 2 out of 4 passed

    def test_pipeline_readiness_with_performance_metrics(self):
        """Test creating PipelineReadiness with performance metrics."""
        check = ValidationCheck(
            name="performance_check",
            description="Check performance metrics",
            success=True
        )

        metrics = PerformanceMetrics(
            id="metrics_1",
            query_latency_ms=50.0,
            success_rate=1.0,
            total_queries=10
        )

        readiness = PipelineReadiness(
            id="readiness_1",
            overall_status="ready",
            checks=[check],
            performance_metrics=metrics
        )

        assert readiness.performance_metrics is not None
        assert readiness.performance_metrics.query_latency_ms == 50.0

    def test_pipeline_readiness_with_recommendations(self):
        """Test creating PipelineReadiness with recommendations."""
        check = ValidationCheck(
            name="recommendation_check",
            description="Check requiring recommendations",
            success=False,
            details={"issue": "high_latency"}
        )

        readiness = PipelineReadiness(
            id="readiness_1",
            overall_status="not_ready",
            checks=[check],
            recommendations=["Optimize query vector size", "Check network connection"]
        )

        assert "Optimize query vector size" in readiness.recommendations
        assert "Check network connection" in readiness.recommendations

    def test_empty_pipeline_readiness(self):
        """Test creating PipelineReadiness with no checks."""
        readiness = PipelineReadiness(
            id="readiness_1",
            overall_status="unknown",
            checks=[]
        )

        assert readiness.passed_checks_count == 0
        assert readiness.failed_checks_count == 0
        assert readiness.success_rate == 0.0