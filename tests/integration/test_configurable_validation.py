"""Integration tests for configurable validation parameters."""

import pytest
from unittest.mock import Mock, patch
from src.services.validation_service import ValidationService
from src.services.qdrant_service import QdrantService
from src.models.query import Query
from src.models.performance_metrics import PerformanceMetrics


class TestConfigurableValidationIntegration:
    """Integration tests for configurable validation parameters."""

    @pytest.fixture
    def mock_qdrant_service(self):
        """Mock Qdrant service for testing."""
        with patch('src.services.qdrant_service.QdrantService') as mock_service:
            yield mock_service

    def test_validate_with_configurable_top_k(self, mock_qdrant_service):
        """Test validation with configurable top-k values."""
        # Create a mock QdrantService
        mock_qdrant = Mock(spec=QdrantService)
        mock_qdrant.search = Mock(return_value=[])

        # Create validation service with mocked Qdrant service
        validation_service = ValidationService(mock_qdrant)

        # Test configurable top-k validation
        top_k_values = [1, 3, 5, 10]
        results = validation_service.validate_with_configurable_top_k("test query", top_k_values)

        # Verify that results contain entries for each top-k value
        assert len(results) == len(top_k_values)
        for k in top_k_values:
            assert k in results
            assert results[k].query.top_k == k

    def test_validate_configurable_retrieval(self, mock_qdrant_service):
        """Test configurable retrieval validation."""
        # Create a mock QdrantService
        mock_qdrant = Mock(spec=QdrantService)
        from src.models.retrieved_chunk import RetrievedChunk
        mock_chunk = RetrievedChunk(id="chunk_1", text="test", score=0.8, position=0)
        mock_qdrant.search = Mock(return_value=[mock_chunk])

        # Create validation service with mocked Qdrant service
        validation_service = ValidationService(mock_qdrant)

        # Create a query with configurable parameters
        query = Query(
            text="test query",
            top_k=5,
            filters={"category": "science"}
        )

        # Execute configurable validation
        result = validation_service.validate_configurable_retrieval(query, "test_log_id")

        # Verify the result
        assert result.success is True
        assert result.query == query
        assert len(result.retrieved_chunks) == 1
        assert result.retrieved_chunks[0].id == "chunk_1"

    def test_validate_top_k_retrieval(self, mock_qdrant_service):
        """Test top-k retrieval validation."""
        # Create a mock QdrantService
        mock_qdrant = Mock(spec=QdrantService)
        from src.models.retrieved_chunk import RetrievedChunk
        # Mock different numbers of results based on the query's top_k
        def mock_search_side_effect(query_obj):
            return [
                RetrievedChunk(id=f"chunk_{i}", text=f"test {i}", score=0.9-i*0.1, position=i)
                for i in range(query_obj.top_k)
            ]
        mock_qdrant.search = Mock(side_effect=mock_search_side_effect)

        # Create validation service with mocked Qdrant service
        validation_service = ValidationService(mock_qdrant)

        # Test with different expected counts
        query = Query(text="test query", top_k=3)
        result = validation_service.validate_top_k_retrieval(query, 3)

        # Verify that the correct number of results were returned
        assert result is True

    def test_validate_relevance_with_configurable_threshold(self, mock_qdrant_service):
        """Test relevance validation with configurable threshold."""
        # Create a mock QdrantService
        mock_qdrant = Mock(spec=QdrantService)
        from src.models.retrieved_chunk import RetrievedChunk
        # Mock results with varying scores
        mock_results = [
            RetrievedChunk(id="chunk_1", text="test 1", score=0.8, position=0),
            RetrievedChunk(id="chunk_2", text="test 2", score=0.6, position=1)
        ]
        mock_qdrant.search = Mock(return_value=mock_results)

        # Create validation service with mocked Qdrant service
        validation_service = ValidationService(mock_qdrant)

        # Test with a threshold that should pass
        query = Query(text="test query", top_k=2)
        result = validation_service.validate_relevance(query, relevance_threshold=0.5)

        # Should pass since highest score (0.8) > threshold (0.5)
        assert result is True

        # Test with a threshold that should fail
        result = validation_service.validate_relevance(query, relevance_threshold=0.9)

        # Should fail since highest score (0.8) < threshold (0.9)
        assert result is False