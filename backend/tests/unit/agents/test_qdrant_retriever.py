"""
Unit tests for Qdrant Retriever Tool
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from qdrant_client import QdrantClient
from backend.src.agents.tools.qdrant_retriever import QdrantRetrieverTool


class TestQdrantRetrieverTool:
    """
    Test cases for QdrantRetrieverTool
    """

    @patch('backend.src.agents.tools.qdrant_retriever.SentenceTransformer')
    @patch('backend.src.agents.tools.qdrant_retriever.QdrantClient')
    def test_init(self, mock_qdrant_client, mock_sentence_transformer):
        """
        Test that QdrantRetrieverTool initializes correctly
        """
        # Arrange
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_embedding_model = Mock()
        mock_sentence_transformer.return_value = mock_embedding_model

        # Act
        tool = QdrantRetrieverTool(host="localhost", port=6333, collection_name="test_collection")

        # Assert
        assert tool.host == "localhost"
        assert tool.port == 6333
        assert tool.collection_name == "test_collection"
        mock_qdrant_client.assert_called_once_with(host="localhost", port=6333)

    @patch('backend.src.agents.tools.qdrant_retriever.SentenceTransformer')
    @patch('backend.src.agents.tools.qdrant_retriever.QdrantClient')
    def test_retrieve_success(self, mock_qdrant_client, mock_sentence_transformer):
        """
        Test that retrieve returns formatted results when successful
        """
        # Arrange
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_embedding_model = Mock()
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_embedding_model.encode.return_value = [0.1, 0.2, 0.3]

        # Mock search results
        mock_hit1 = Mock()
        mock_hit1.payload = {"content": "This is content 1", "metadata": {"source": "doc1"}}
        mock_hit1.score = 0.9
        mock_hit2 = Mock()
        mock_hit2.payload = {"content": "This is content 2", "metadata": {"source": "doc2"}}
        mock_hit2.score = 0.8

        mock_client_instance.search.return_value = [mock_hit1, mock_hit2]

        tool = QdrantRetrieverTool(collection_name="test_collection")

        # Act
        results = tool.retrieve("test query", top_k=2)

        # Assert
        assert len(results) == 2
        assert results[0]["content"] == "This is content 1"
        assert results[0]["metadata"]["source"] == "doc1"
        assert results[0]["score"] == 0.9
        assert results[1]["content"] == "This is content 2"
        assert results[1]["metadata"]["source"] == "doc2"
        assert results[1]["score"] == 0.8

    @patch('backend.src.agents.tools.qdrant_retriever.SentenceTransformer')
    @patch('backend.src.agents.tools.qdrant_retriever.QdrantClient')
    def test_retrieve_with_exception(self, mock_qdrant_client, mock_sentence_transformer):
        """
        Test that retrieve handles exceptions properly
        """
        # Arrange
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_embedding_model = Mock()
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_embedding_model.encode.return_value = [0.1, 0.2, 0.3]

        mock_client_instance.search.side_effect = Exception("Connection failed")

        tool = QdrantRetrieverTool(collection_name="test_collection")

        # Act & Assert
        with pytest.raises(Exception, match="Error retrieving from Qdrant: Connection failed"):
            tool.retrieve("test query", top_k=2)

    @patch('backend.src.agents.tools.qdrant_retriever.SentenceTransformer')
    @patch('backend.src.agents.tools.qdrant_retriever.QdrantClient')
    def test_validate_connection_success(self, mock_qdrant_client, mock_sentence_transformer):
        """
        Test that validate_connection returns True when connection is successful
        """
        # Arrange
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_embedding_model = Mock()
        mock_sentence_transformer.return_value = mock_embedding_model

        # Mock the get_collections method
        mock_client_instance.get_collections.return_value = Mock()

        tool = QdrantRetrieverTool()

        # Act
        result = tool.validate_connection()

        # Assert
        assert result is True

    @patch('backend.src.agents.tools.qdrant_retriever.SentenceTransformer')
    @patch('backend.src.agents.tools.qdrant_retriever.QdrantClient')
    def test_validate_connection_failure(self, mock_qdrant_client, mock_sentence_transformer):
        """
        Test that validate_connection returns False when connection fails
        """
        # Arrange
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_embedding_model = Mock()
        mock_sentence_transformer.return_value = mock_embedding_model

        # Mock the get_collections method to raise an exception
        mock_client_instance.get_collections.side_effect = Exception("Connection failed")

        tool = QdrantRetrieverTool()

        # Act
        result = tool.validate_connection()

        # Assert
        assert result is False

    @patch('backend.src.agents.tools.qdrant_retriever.SentenceTransformer')
    @patch('backend.src.agents.tools.qdrant_retriever.QdrantClient')
    def test_validate_collection_exists_success(self, mock_qdrant_client, mock_sentence_transformer):
        """
        Test that validate_collection_exists returns True when collection exists
        """
        # Arrange
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_embedding_model = Mock()
        mock_sentence_transformer.return_value = mock_embedding_model

        # Mock the get_collection method
        mock_client_instance.get_collection.return_value = Mock()

        tool = QdrantRetrieverTool(collection_name="existing_collection")

        # Act
        result = tool.validate_collection_exists()

        # Assert
        assert result is True

    @patch('backend.src.agents.tools.qdrant_retriever.SentenceTransformer')
    @patch('backend.src.agents.tools.qdrant_retriever.QdrantClient')
    def test_validate_collection_exists_failure(self, mock_qdrant_client, mock_sentence_transformer):
        """
        Test that validate_collection_exists returns False when collection doesn't exist
        """
        # Arrange
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_embedding_model = Mock()
        mock_sentence_transformer.return_value = mock_embedding_model

        # Mock the get_collection method to raise an exception
        mock_client_instance.get_collection.side_effect = Exception("Collection not found")

        tool = QdrantRetrieverTool(collection_name="nonexistent_collection")

        # Act
        result = tool.validate_collection_exists()

        # Assert
        assert result is False

    @patch('backend.src.agents.tools.qdrant_retriever.SentenceTransformer')
    @patch('backend.src.agents.tools.qdrant_retriever.QdrantClient')
    def test_get_content_for_agent_with_results(self, mock_qdrant_client, mock_sentence_transformer):
        """
        Test that get_content_for_agent returns formatted content when results are found
        """
        # Arrange
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_embedding_model = Mock()
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_embedding_model.encode.return_value = [0.1, 0.2, 0.3]

        # Mock search results
        mock_hit1 = Mock()
        mock_hit1.payload = {"content": "This is content 1", "metadata": {"source": "doc1"}}
        mock_hit1.score = 0.9

        mock_client_instance.search.return_value = [mock_hit1]

        tool = QdrantRetrieverTool(collection_name="test_collection")

        # Act
        content = tool.get_content_for_agent("test query", top_k=1)

        # Assert
        assert "Retrieved content for answering the question:" in content
        assert "This is content 1" in content
        assert "doc1" in content

    @patch('backend.src.agents.tools.qdrant_retriever.SentenceTransformer')
    @patch('backend.src.agents.tools.qdrant_retriever.QdrantClient')
    def test_get_content_for_agent_no_results(self, mock_qdrant_client, mock_sentence_transformer):
        """
        Test that get_content_for_agent returns appropriate message when no results found
        """
        # Arrange
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_embedding_model = Mock()
        mock_sentence_transformer.return_value = mock_embedding_model
        mock_embedding_model.encode.return_value = [0.1, 0.2, 0.3]

        # Mock empty search results
        mock_client_instance.search.return_value = []

        tool = QdrantRetrieverTool(collection_name="test_collection")

        # Act
        content = tool.get_content_for_agent("test query", top_k=1)

        # Assert
        assert content == "No relevant content found in the knowledge base."