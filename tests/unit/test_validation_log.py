"""Unit tests for ValidationLog model."""

import pytest
from datetime import datetime
from src.models.validation_log import ValidationLog
from src.models.query import Query
from src.models.retrieved_chunk import RetrievedChunk


class TestValidationLogModel:
    """Test cases for the ValidationLog model."""

    def test_validation_log_creation_valid(self):
        """Test creating a valid ValidationLog instance."""
        query = Query(text="test query", top_k=5)
        chunk = RetrievedChunk(id="chunk_1", text="test content", score=0.85, position=0)

        log = ValidationLog(
            id="log_1",
            query=query,
            execution_time_ms=100.5,
            success=True,
            retrieved_chunks=[chunk]
        )

        assert log.id == "log_1"
        assert log.query == query
        assert log.execution_time_ms == 100.5
        assert log.success is True
        assert log.retrieved_chunks == [chunk]
        assert isinstance(log.timestamp, datetime)

    def test_validation_log_with_error(self):
        """Test creating a ValidationLog with an error message."""
        query = Query(text="test query", top_k=5)

        log = ValidationLog(
            id="log_1",
            query=query,
            execution_time_ms=50.0,
            success=False,
            error_message="Connection timeout"
        )

        assert log.success is False
        assert log.error_message == "Connection timeout"

    def test_validation_log_execution_time_validation(self):
        """Test that execution_time_ms is validated correctly."""
        query = Query(text="test query", top_k=5)

        # Valid execution time
        ValidationLog(id="log_1", query=query, execution_time_ms=0.0, success=True)
        ValidationLog(id="log_1", query=query, execution_time_ms=100.5, success=True)

        # Invalid execution time should raise validation error
        with pytest.raises(ValueError):
            ValidationLog(id="log_1", query=query, execution_time_ms=-1.0, success=True)

    def test_validation_log_with_metadata(self):
        """Test creating a ValidationLog with metadata."""
        query = Query(text="test query", top_k=5)
        chunk = RetrievedChunk(id="chunk_1", text="test content", score=0.85, position=0)

        metadata = {"user_id": "test_user", "validation_type": "semantic_search"}

        log = ValidationLog(
            id="log_1",
            query=query,
            retrieved_chunks=[chunk],
            execution_time_ms=100.5,
            success=True,
            metadata=metadata
        )

        assert log.metadata == metadata

    def test_validation_log_defaults(self):
        """Test ValidationLog with default values."""
        query = Query(text="test query", top_k=5)

        log = ValidationLog(
            id="log_1",
            query=query,
            execution_time_ms=100.5,
            success=True
        )

        # Should have empty retrieved_chunks by default
        assert log.retrieved_chunks == []
        # Should have empty metadata by default
        assert log.metadata == {}