"""Integration tests for full pipeline validation."""

import pytest
from unittest.mock import Mock, patch
from src.services.validation_service import ValidationService
from src.services.qdrant_service import QdrantService
from src.models.query import Query
from src.models.retrieved_chunk import RetrievedChunk
from src.models.pipeline_readiness import PipelineReadiness


class TestPipelineValidationIntegration:
    """Integration tests for full pipeline validation."""

    @pytest.fixture
    def mock_services(self):
        """Mock Qdrant and validation services for testing."""
        with patch('src.services.qdrant_service.QdrantService') as mock_qdrant, \
             patch('src.services.validation_service.ValidationService') as mock_validation:
            yield mock_qdrant, mock_validation

    def test_comprehensive_pipeline_validation(self):
        """Test comprehensive pipeline validation."""
        # Create mock QdrantService
        mock_qdrant = Mock(spec=QdrantService)
        mock_chunk = RetrievedChunk(id="chunk_1", text="test content", score=0.85, position=0)
        mock_qdrant.search = Mock(return_value=[mock_chunk])
        mock_qdrant.health_check = Mock(return_value=True)

        # Create validation service with mocked Qdrant service
        validation_service = ValidationService(mock_qdrant)

        # Perform a basic validation to test the pipeline
        query = Query(text="test query", top_k=5)
        log_id = "pipeline_test_1"
        validation_log = validation_service.validate_retrieval(query, log_id)

        # Verify the validation was successful
        assert validation_log.success is True
        assert len(validation_log.retrieved_chunks) == 1
        assert validation_log.retrieved_chunks[0].id == "chunk_1"

    def test_pipeline_readiness_check(self):
        """Test pipeline readiness checking functionality."""
        # Create mock QdrantService
        mock_qdrant = Mock(spec=QdrantService)
        mock_chunk = RetrievedChunk(id="chunk_1", text="test content", score=0.85, position=0)
        mock_qdrant.search = Mock(return_value=[mock_chunk])
        mock_qdrant.health_check = Mock(return_value=True)

        # Create validation service with mocked Qdrant service
        validation_service = ValidationService(mock_qdrant)

        # Test the pipeline readiness by performing multiple checks
        checks = []

        # Check 1: Health check
        health_ok = mock_qdrant.health_check()
        checks.append({
            "name": "health_check",
            "description": "Check if Qdrant service is healthy",
            "success": health_ok
        })

        # Check 2: Basic retrieval
        query = Query(text="test query", top_k=1)
        validation_log = validation_service.validate_retrieval(query, "readiness_test")
        checks.append({
            "name": "basic_retrieval",
            "description": "Test basic retrieval functionality",
            "success": validation_log.success
        })

        # Check 3: Top-k validation
        top_k_valid = validation_service.validate_top_k_retrieval(query, 1)
        checks.append({
            "name": "top_k_validation",
            "description": "Validate top-k retrieval works correctly",
            "success": top_k_valid
        })

        # Verify all checks passed
        assert all(check["success"] for check in checks)

    def test_pipeline_with_failure_scenario(self):
        """Test pipeline validation when some components fail."""
        # Create mock QdrantService that will fail on search
        mock_qdrant = Mock(spec=QdrantService)
        mock_qdrant.search = Mock(side_effect=Exception("Connection failed"))
        mock_qdrant.health_check = Mock(return_value=True)

        # Create validation service with mocked Qdrant service
        validation_service = ValidationService(mock_qdrant)

        # Try to perform validation - should fail
        query = Query(text="test query", top_k=5)
        validation_log = validation_service.validate_retrieval(query, "failure_test")

        # Verify the validation failed as expected
        assert validation_log.success is False
        assert "Connection failed" in validation_log.error_message

    def test_pipeline_relevance_validation(self):
        """Test pipeline validation for relevance checking."""
        # Create mock QdrantService with results that have varying scores
        mock_qdrant = Mock(spec=QdrantService)
        high_score_chunk = RetrievedChunk(id="chunk_1", text="relevant content", score=0.9, position=0)
        low_score_chunk = RetrievedChunk(id="chunk_2", text="less relevant", score=0.3, position=1)
        mock_qdrant.search = Mock(return_value=[high_score_chunk, low_score_chunk])
        mock_qdrant.health_check = Mock(return_value=True)

        # Create validation service with mocked Qdrant service
        validation_service = ValidationService(mock_qdrant)

        # Test relevance validation with different thresholds
        query = Query(text="test query", top_k=2)

        # Should pass with low threshold
        relevance_ok = validation_service.validate_relevance(query, relevance_threshold=0.5)
        assert relevance_ok is True  # Because highest score is 0.9

        # Should fail with high threshold
        relevance_ok = validation_service.validate_relevance(query, relevance_threshold=0.95)
        assert relevance_ok is False  # Because highest score is 0.9

    def test_pipeline_with_configurable_params(self):
        """Test pipeline validation with configurable parameters."""
        # Create mock QdrantService
        mock_qdrant = Mock(spec=QdrantService)
        mock_chunk = RetrievedChunk(id="chunk_1", text="test content", score=0.8, position=0)
        mock_qdrant.search = Mock(return_value=[mock_chunk])
        mock_qdrant.health_check = Mock(return_value=True)

        # Create validation service with mocked Qdrant service
        validation_service = ValidationService(mock_qdrant)

        # Test configurable validation
        query = Query(text="test query", top_k=3)
        validation_log = validation_service.validate_configurable_retrieval(query, "config_test")

        # Should be successful
        assert validation_log.success is True
        assert validation_log.query.top_k == 3