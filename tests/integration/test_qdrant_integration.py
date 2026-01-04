"""Integration tests for Qdrant service."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.services.qdrant_service import QdrantService
from src.models.query import Query
from src.models.retrieved_chunk import RetrievedChunk


class TestQdrantServiceIntegration:
    """Integration tests for the Qdrant service."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client for testing."""
        with patch('qdrant_client.QdrantClient') as mock_client:
            yield mock_client

    def test_qdrant_service_initialization(self, mock_qdrant_client):
        """Test initializing the Qdrant service."""
        from src.lib.config import QdrantConfig

        config = QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="test_collection"
        )

        service = QdrantService(config)

        assert service.config == config
        assert service.client is not None

    def test_search_method_with_mock(self, mock_qdrant_client):
        """Test the search method with mocked Qdrant client."""
        from src.lib.config import QdrantConfig

        config = QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="test_collection"
        )

        # Create service instance
        service = QdrantService(config)

        # Mock the search response
        mock_result = [
            MagicMock(
                id="chunk_1",
                payload={"text": "test content", "source": "doc1.pdf"},
                score=0.85
            )
        ]

        # Patch the client's search method
        service.client.search = Mock(return_value=mock_result)

        query = Query(text="test query", top_k=5)
        results = service.search(query)

        # Verify the results
        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].id == "chunk_1"
        assert results[0].text == "test content"
        assert results[0].score == 0.85
        assert results[0].metadata == {"source": "doc1.pdf"}

    def test_search_with_filters(self, mock_qdrant_client):
        """Test the search method with filters."""
        from src.lib.config import QdrantConfig

        config = QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="test_collection"
        )

        # Create service instance
        service = QdrantService(config)

        # Mock the search response
        mock_result = [
            MagicMock(
                id="chunk_2",
                payload={"text": "filtered content", "category": "science"},
                score=0.90
            )
        ]

        # Patch the client's search method
        service.client.search = Mock(return_value=mock_result)

        query = Query(text="test query", top_k=5, filters={"category": "science"})
        results = service.search(query)

        # Verify the results
        assert len(results) == 1
        assert results[0].id == "chunk_2"
        assert results[0].text == "filtered content"
        assert results[0].score == 0.90
        assert results[0].metadata == {"category": "science"}

    def test_search_with_vector(self, mock_qdrant_client):
        """Test the search method with a pre-computed vector."""
        from src.lib.config import QdrantConfig

        config = QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="test_collection"
        )

        # Create service instance
        service = QdrantService(config)

        # Mock the search response
        mock_result = [
            MagicMock(
                id="chunk_3",
                payload={"text": "vector search result", "source": "doc3.pdf"},
                score=0.75
            )
        ]

        # Patch the client's search method
        service.client.search = Mock(return_value=mock_result)

        query = Query(text="test query", top_k=5, query_vector=[0.1, 0.2, 0.3])
        results = service.search(query)

        # Verify the results
        assert len(results) == 1
        assert results[0].id == "chunk_3"
        assert results[0].text == "vector search result"
        assert results[0].score == 0.75