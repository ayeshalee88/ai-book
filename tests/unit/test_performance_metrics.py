"""Unit tests for PerformanceMetrics model."""

import pytest
from datetime import datetime
from src.models.performance_metrics import PerformanceMetrics


class TestPerformanceMetricsModel:
    """Test cases for the PerformanceMetrics model."""

    def test_performance_metrics_creation_valid(self):
        """Test creating a valid PerformanceMetrics instance."""
        metrics = PerformanceMetrics(
            id="metrics_1",
            query_latency_ms=100.5,
            success_rate=0.95,
            total_queries=100
        )
        assert metrics.id == "metrics_1"
        assert metrics.query_latency_ms == 100.5
        assert metrics.success_rate == 0.95
        assert metrics.total_queries == 100
        assert isinstance(metrics.timestamp, datetime)

    def test_performance_metrics_with_optional_fields(self):
        """Test creating PerformanceMetrics with optional fields."""
        metrics = PerformanceMetrics(
            id="metrics_1",
            query_latency_ms=50.0,
            p95_latency_ms=120.0,
            p99_latency_ms=200.0,
            throughput_qps=10.5,
            success_rate=1.0,
            total_queries=50,
            cache_hit_rate=0.8,
            memory_usage_mb=128.5
        )
        assert metrics.p95_latency_ms == 120.0
        assert metrics.p99_latency_ms == 200.0
        assert metrics.throughput_qps == 10.5
        assert metrics.cache_hit_rate == 0.8
        assert metrics.memory_usage_mb == 128.5

    def test_performance_metrics_rate_validation(self):
        """Test that success_rate and error_rate are validated correctly."""
        # Valid rates
        PerformanceMetrics(id="metrics_1", query_latency_ms=100.0, success_rate=0.0, total_queries=0)
        PerformanceMetrics(id="metrics_1", query_latency_ms=100.0, success_rate=1.0, total_queries=0)
        PerformanceMetrics(id="metrics_1", query_latency_ms=100.0, error_rate=0.0, total_queries=0)
        PerformanceMetrics(id="metrics_1", query_latency_ms=100.0, error_rate=1.0, total_queries=0)

        # Invalid rates should raise validation error
        with pytest.raises(ValueError):
            PerformanceMetrics(id="metrics_1", query_latency_ms=100.0, success_rate=-0.1, total_queries=0)

        with pytest.raises(ValueError):
            PerformanceMetrics(id="metrics_1", query_latency_ms=100.0, success_rate=1.1, total_queries=0)

        with pytest.raises(ValueError):
            PerformanceMetrics(id="metrics_1", query_latency_ms=100.0, error_rate=-0.1, total_queries=0)

        with pytest.raises(ValueError):
            PerformanceMetrics(id="metrics_1", query_latency_ms=100.0, error_rate=1.1, total_queries=0)

    def test_performance_metrics_latency_validation(self):
        """Test that latency values are validated correctly."""
        # Valid latency
        PerformanceMetrics(id="metrics_1", query_latency_ms=0.0, success_rate=1.0, total_queries=0)
        PerformanceMetrics(id="metrics_1", query_latency_ms=100.5, success_rate=1.0, total_queries=0)

        # Invalid latency should raise validation error
        with pytest.raises(ValueError):
            PerformanceMetrics(id="metrics_1", query_latency_ms=-1.0, success_rate=1.0, total_queries=0)

    def test_performance_metrics_query_count_validation(self):
        """Test that total_queries is validated correctly."""
        # Valid query counts
        PerformanceMetrics(id="metrics_1", query_latency_ms=100.0, success_rate=1.0, total_queries=0)
        PerformanceMetrics(id="metrics_1", query_latency_ms=100.0, success_rate=1.0, total_queries=100)

        # Invalid query count should raise validation error
        with pytest.raises(ValueError):
            PerformanceMetrics(id="metrics_1", query_latency_ms=100.0, success_rate=1.0, total_queries=-1)

    def test_performance_metrics_with_additional_metrics(self):
        """Test creating PerformanceMetrics with additional custom metrics."""
        additional = {"cpu_usage": 0.75, "disk_io": 1000}
        metrics = PerformanceMetrics(
            id="metrics_1",
            query_latency_ms=100.0,
            success_rate=1.0,
            total_queries=10,
            additional_metrics=additional
        )
        assert metrics.additional_metrics == additional