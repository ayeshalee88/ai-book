"""
Unit tests for QA API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from backend.src.api.main import app
from backend.src.agents.rag_agent import RAGAgent


class TestQAEndpoints:
    """
    Test cases for QA API endpoints
    """

    def setup_method(self):
        """
        Setup method to create test client
        """
        self.client = TestClient(app)

    @patch('backend.src.api.endpoints.qa.rag_agent')
    def test_ask_question_success(self, mock_rag_agent):
        """
        Test that the /qa endpoint returns a proper response for a valid question
        """
        # Arrange
        mock_rag_agent.answer_question.return_value = {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
            "sources": [
                {
                    "content_snippet": "France is a country in Europe. Paris is its capital.",
                    "score": 0.9,
                    "source_metadata": {"source": "wikipedia", "page": 1}
                }
            ],
            "retrieved_chunks_count": 1
        }

        # Act
        response = self.client.post(
            "/api/v1/qa",
            json={
                "question": "What is the capital of France?",
                "top_k": 3
            }
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What is the capital of France?"
        assert data["answer"] == "The capital of France is Paris."
        assert len(data["sources"]) == 1
        assert data["retrieved_chunks_count"] == 1

    @patch('backend.src.api.endpoints.qa.rag_agent')
    def test_ask_question_with_error(self, mock_rag_agent):
        """
        Test that the /qa endpoint handles errors gracefully
        """
        # Arrange
        mock_rag_agent.answer_question.side_effect = Exception("Processing failed")

        # Act
        response = self.client.post(
            "/api/v1/qa",
            json={
                "question": "What is the capital of Mars?",
                "top_k": 3
            }
        )

        # Assert
        assert response.status_code == 500
        assert "Error processing question" in response.json()["detail"]

    def test_health_check_success(self):
        """
        Test that the /health endpoint returns proper status when services are available
        """
        # Since we're testing the real endpoint, we'll mock the RAG agent validation
        # at the agent level to return success
        with patch('backend.src.api.endpoints.qa.rag_agent') as mock_rag_agent:
            mock_rag_agent.validate_setup.return_value = {
                "llm_service": True,
                "retriever_service": True,
                "overall": True
            }

            # Act
            response = self.client.get("/api/v1/health")

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["services"]["overall"] is True

    def test_health_check_failure(self):
        """
        Test that the /health endpoint returns proper status when services are unavailable
        """
        # Since we're testing the real endpoint, we'll mock the RAG agent validation
        # at the agent level to return failure
        with patch('backend.src.api.endpoints.qa.rag_agent') as mock_rag_agent:
            mock_rag_agent.validate_setup.return_value = {
                "llm_service": False,
                "retriever_service": True,
                "overall": False
            }

            # Act
            response = self.client.get("/api/v1/health")

            # Assert
            assert response.status_code == 200  # Health check itself doesn't fail
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["services"]["overall"] is False

    def test_health_check_exception(self):
        """
        Test that the /health endpoint handles exceptions gracefully
        """
        # Since we're testing the real endpoint, we'll mock the RAG agent validation
        # at the agent level to raise an exception
        with patch('backend.src.api.endpoints.qa.rag_agent') as mock_rag_agent:
            mock_rag_agent.validate_setup.side_effect = Exception("Health check failed")

            # Act
            response = self.client.get("/api/v1/health")

            # Assert
            assert response.status_code == 200  # Health check returns unhealthy, not error
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["services"]["llm_service"] is False
            assert data["services"]["retriever_service"] is False
            assert data["services"]["overall"] is False

    def test_validate_setup_success(self):
        """
        Test that the /validate endpoint returns proper validation status
        """
        # Since we're testing the real endpoint, we'll mock the RAG agent validation
        with patch('backend.src.api.endpoints.qa.rag_agent') as mock_rag_agent:
            mock_rag_agent.validate_setup.return_value = {
                "llm_service": True,
                "retriever_service": True,
                "overall": True
            }

            # Act
            response = self.client.get("/api/v1/validate")

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["validation_result"]["overall"] is True
            assert "message" in data
            assert data["message"] == "Setup validation completed"

    def test_validate_setup_exception(self):
        """
        Test that the /validate endpoint handles exceptions gracefully
        """
        # Since we're testing the real endpoint, we'll mock the RAG agent validation
        with patch('backend.src.api.endpoints.qa.rag_agent') as mock_rag_agent:
            mock_rag_agent.validate_setup.side_effect = Exception("Validation failed")

            # Act
            response = self.client.get("/api/v1/validate")

            # Assert
            assert response.status_code == 500
            assert "Validation failed" in response.json()["detail"]

    def test_root_endpoint(self):
        """
        Test that the root endpoint returns proper service information
        """
        # Act
        response = self.client.get("/")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "RAG Chatbot - Agent-Based QA Service" in data["message"]
        assert "endpoints" in data
        assert len(data["endpoints"]) > 0