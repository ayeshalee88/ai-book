"""
Integration tests for the complete QA flow
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from backend.src.api.main import app
from backend.src.agents.rag_agent import RAGAgent
from backend.src.services.llm_service import OpenRouterService
from backend.src.agents.tools.qdrant_retriever import QdrantRetrieverTool


class TestQAIntegration:
    """
    Integration tests for the complete QA flow
    """

    def setup_method(self):
        """
        Setup method to create test client
        """
        self.client = TestClient(app)

    def test_complete_qa_flow_success(self):
        """
        Test the complete QA flow from API request to response with mocked services
        """
        # Mock the RAG agent and its dependencies to simulate a successful flow
        with patch('backend.src.api.endpoints.qa.RAGAgent') as MockRAGAgent:
            # Create a mock instance
            mock_agent_instance = Mock()

            # Mock the answer_question method to return a successful response
            mock_agent_instance.answer_question.return_value = {
                "question": "What is the capital of France?",
                "answer": "The capital of France is Paris, which is located in the north-central part of the country and serves as the political and cultural center.",
                "sources": [
                    {
                        "content_snippet": "France is a country in Europe. Paris is its capital and largest city.",
                        "score": 0.95,
                        "source_metadata": {
                            "source": "wikipedia",
                            "title": "France",
                            "url": "https://en.wikipedia.org/wiki/France",
                            "section": "Geography"
                        }
                    },
                    {
                        "content_snippet": "Paris is the capital and most populous city of France. It has been one of Europe's major centers of finance, diplomacy, commerce, and arts.",
                        "score": 0.89,
                        "source_metadata": {
                            "source": "encyclopedia",
                            "title": "Paris",
                            "volume": "Volume 3",
                            "page": 42
                        }
                    }
                ],
                "retrieved_chunks_count": 2,
                "error": None
            }

            # Configure the mock class to return our mock instance
            MockRAGAgent.return_value = mock_agent_instance

            # Act: Make a request to the QA endpoint
            response = self.client.post(
                "/api/v1/qa",
                json={
                    "question": "What is the capital of France?",
                    "top_k": 5
                }
            )

            # Assert: Check that the response is successful
            assert response.status_code == 200

            # Parse the response
            data = response.json()

            # Verify the response structure and content
            assert data["question"] == "What is the capital of France?"
            assert "Paris" in data["answer"]
            assert len(data["sources"]) == 2
            assert data["retrieved_chunks_count"] == 2
            assert "error" not in data or data.get("error") is None

            # Verify source information is properly formatted
            first_source = data["sources"][0]
            assert "content_snippet" in first_source
            assert "score" in first_source
            assert "source_metadata" in first_source
            assert first_source["score"] > 0.8

    def test_complete_qa_flow_with_no_content_found(self):
        """
        Test the QA flow when no relevant content is found in the knowledge base
        """
        # Mock the RAG agent to simulate no content found scenario
        with patch('backend.src.api.endpoints.qa.RAGAgent') as MockRAGAgent:
            # Create a mock instance
            mock_agent_instance = Mock()

            # Mock the answer_question method to return a response with no sources
            mock_agent_instance.answer_question.return_value = {
                "question": "What is the capital of Mars?",
                "answer": "I cannot find information about the capital of Mars in the knowledge base. Mars does not have a capital as it is not a political entity with a government.",
                "sources": [],
                "retrieved_chunks_count": 0,
                "error": None
            }

            # Configure the mock class to return our mock instance
            MockRAGAgent.return_value = mock_agent_instance

            # Act: Make a request to the QA endpoint
            response = self.client.post(
                "/api/v1/qa",
                json={
                    "question": "What is the capital of Mars?",
                    "top_k": 5
                }
            )

            # Assert: Check that the response is successful but with no sources
            assert response.status_code == 200

            # Parse the response
            data = response.json()

            # Verify the response structure and content
            assert data["question"] == "What is the capital of Mars?"
            assert "Mars" in data["answer"]
            assert len(data["sources"]) == 0
            assert data["retrieved_chunks_count"] == 0
            assert "error" not in data or data.get("error") is None

    def test_complete_qa_flow_with_error_handling(self):
        """
        Test the QA flow error handling when services fail
        """
        # Mock the RAG agent to simulate an error in processing
        with patch('backend.src.api.endpoints.qa.RAGAgent') as MockRAGAgent:
            # Create a mock instance
            mock_agent_instance = Mock()

            # Mock the answer_question method to raise an exception
            mock_agent_instance.answer_question.side_effect = Exception("Service temporarily unavailable")

            # Configure the mock class to return our mock instance
            MockRAGAgent.return_value = mock_agent_instance

            # Act: Make a request to the QA endpoint
            response = self.client.post(
                "/api/v1/qa",
                json={
                    "question": "What is the meaning of life?",
                    "top_k": 3
                }
            )

            # Assert: Check that the response is an error
            assert response.status_code == 500
            assert "Error processing question" in response.json()["detail"]

    def test_health_check_integration(self):
        """
        Test the health check endpoint integration
        """
        # Mock the RAG agent validation to return healthy status
        with patch('backend.src.api.endpoints.qa.RAGAgent') as MockRAGAgent:
            # Create a mock instance
            mock_agent_instance = Mock()

            # Mock the validate_setup method to return healthy status
            mock_agent_instance.validate_setup.return_value = {
                "llm_service": True,
                "retriever_service": True,
                "overall": True
            }

            # Configure the mock class to return our mock instance
            MockRAGAgent.return_value = mock_agent_instance

            # Act: Make a request to the health endpoint
            response = self.client.get("/api/v1/health")

            # Assert: Check that the response indicates healthy status
            assert response.status_code == 200

            # Parse the response
            data = response.json()

            # Verify the response structure and content
            assert data["status"] == "healthy"
            assert data["services"]["overall"] is True
            assert data["services"]["llm_service"] is True
            assert data["services"]["retriever_service"] is True

    def test_validation_endpoint_integration(self):
        """
        Test the validation endpoint integration
        """
        # Mock the RAG agent validation to return validation results
        with patch('backend.src.api.endpoints.qa.RAGAgent') as MockRAGAgent:
            # Create a mock instance
            mock_agent_instance = Mock()

            # Mock the validate_setup method to return validation results
            mock_agent_instance.validate_setup.return_value = {
                "llm_service": True,
                "retriever_service": True,
                "overall": True
            }

            # Configure the mock class to return our mock instance
            MockRAGAgent.return_value = mock_agent_instance

            # Act: Make a request to the validation endpoint
            response = self.client.get("/api/v1/validate")

            # Assert: Check that the response is successful
            assert response.status_code == 200

            # Parse the response
            data = response.json()

            # Verify the response structure and content
            assert "validation_result" in data
            assert data["validation_result"]["overall"] is True
            assert "message" in data
            assert data["message"] == "Setup validation completed"