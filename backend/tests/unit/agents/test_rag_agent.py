"""
Unit tests for RAG Agent
"""
import pytest
from unittest.mock import Mock, patch
from backend.src.agents.rag_agent import RAGAgent
from backend.src.services.llm_service import OpenRouterService
from backend.src.agents.tools.qdrant_retriever import QdrantRetrieverTool


class TestRAGAgent:
    """
    Test cases for the RAG Agent
    """

    def setup_method(self):
        """
        Setup method to create a mock RAG agent for testing
        """
        # Create a RAG agent with mocked dependencies
        self.agent = RAGAgent()

        # Replace the real services with mocks
        self.mock_llm_service = Mock(spec=OpenRouterService)
        self.mock_retriever_tool = Mock(spec=QdrantRetrieverTool)

        self.agent.llm_service = self.mock_llm_service
        self.agent.retriever_tool = self.mock_retriever_tool

    def test_answer_question_success(self):
        """
        Test that answer_question returns a proper response when services work correctly
        """
        # Arrange
        question = "What is the capital of France?"
        mock_answer = "The capital of France is Paris."
        mock_content = "France is a country in Europe. Paris is its capital."
        mock_sources = [
            {
                "content_snippet": "France is a country in Europe. Paris is its capital.",
                "score": 0.9,
                "source_metadata": {"source": "wikipedia", "page": 1}
            }
        ]

        # Configure mocks
        self.mock_retriever_tool.get_content_for_agent.return_value = mock_content
        self.mock_llm_service.generate_response.return_value = mock_answer
        self.mock_retriever_tool.retrieve.return_value = [
            {
                "content": "France is a country in Europe. Paris is its capital.",
                "metadata": {"source": "wikipedia", "page": 1},
                "score": 0.9
            }
        ]

        # Act
        result = self.agent.answer_question(question, top_k=3)

        # Assert
        assert result["question"] == question
        assert result["answer"] == mock_answer
        assert result["retrieved_chunks_count"] == 1
        assert len(result["sources"]) == 1
        assert "error" not in result or result.get("error") is None

    def test_answer_question_with_error(self):
        """
        Test that answer_question handles errors gracefully
        """
        # Arrange
        question = "What is the capital of Mars?"

        # Configure mocks to raise an exception
        self.mock_retriever_tool.get_content_for_agent.side_effect = Exception("Connection failed")

        # Act
        result = self.agent.answer_question(question, top_k=3)

        # Assert
        assert result["question"] == question
        assert "error" in result
        assert result["answer"].startswith("Error processing question")
        assert result["retrieved_chunks_count"] == 0
        assert len(result["sources"]) == 0

    def test_extract_sources_success(self):
        """
        Test that _extract_sources properly formats source information
        """
        # Arrange
        question = "Test question"
        mock_raw_results = [
            {
                "content": "This is a long content that should be truncated to prevent overly long snippets in the response. " * 5,
                "metadata": {"source": "book", "chapter": 3, "page": 42},
                "score": 0.85
            },
            {
                "content": "Short content",
                "metadata": {"source": "article", "author": "Smith"},
                "score": 0.72
            }
        ]

        # Configure mock
        self.mock_retriever_tool.retrieve.return_value = mock_raw_results

        # Act
        sources = self.agent._extract_sources(question, top_k=2)

        # Assert
        assert len(sources) == 2
        assert sources[0]["score"] == 0.85
        assert sources[0]["source_metadata"]["source"] == "book"
        assert sources[1]["score"] == 0.72
        assert sources[1]["source_metadata"]["author"] == "Smith"
        # Check that long content is truncated
        assert len(sources[0]["content_snippet"]) <= 203  # 200 + "..."
        assert sources[0]["content_snippet"].endswith("...")

    def test_validate_setup_success(self):
        """
        Test that validate_setup returns correct status when all services are available
        """
        # Arrange
        self.mock_llm_service.validate_connection.return_value = True
        self.mock_retriever_tool.validate_connection.return_value = True

        # Act
        result = self.agent.validate_setup()

        # Assert
        assert result["llm_service"] is True
        assert result["retriever_service"] is True
        assert result["overall"] is True

    def test_validate_setup_failure(self):
        """
        Test that validate_setup returns correct status when one service is unavailable
        """
        # Arrange
        self.mock_llm_service.validate_connection.return_value = False
        self.mock_retriever_tool.validate_connection.return_value = True

        # Act
        result = self.agent.validate_setup()

        # Assert
        assert result["llm_service"] is False
        assert result["retriever_service"] is True
        assert result["overall"] is False