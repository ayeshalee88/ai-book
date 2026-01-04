"""
Acceptance scenario tests for User Story 1
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from backend.src.api.main import app


class TestAcceptanceScenarios:
    """
    Acceptance scenario tests for User Story 1 - Query Question Answering
    """

    def setup_method(self):
        """
        Setup method to create test client
        """
        self.client = TestClient(app)

    def test_acceptance_scenario_1_valid_question_returns_answer_with_citations(self):
        """
        Acceptance scenario 1: Valid question returns grounded answer with citations
        """
        # Mock the RAG agent to simulate a successful response with citations
        with patch('backend.src.api.endpoints.qa.RAGAgent') as MockRAGAgent:
            # Create a mock instance
            mock_agent_instance = Mock()

            # Mock the answer_question method to return a response with sources
            mock_agent_instance.answer_question.return_value = {
                "question": "What is the capital of France?",
                "answer": "The capital of France is Paris, which is located in the north-central part of the country.",
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
                    }
                ],
                "retrieved_chunks_count": 1,
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

            # Assert: Check that the response meets acceptance criteria
            assert response.status_code == 200

            # Parse the response
            data = response.json()

            # Verify that a grounded answer was returned
            assert "Paris" in data["answer"]
            assert len(data["answer"]) > 0  # Non-empty answer
            assert data["answer"] != "I don't know"  # Proper answer, not a "don't know" response

            # Verify that citations were provided
            assert len(data["sources"]) > 0  # At least one source provided
            first_source = data["sources"][0]
            assert "content_snippet" in first_source
            assert "score" in first_source
            assert "source_metadata" in first_source
            assert len(first_source["content_snippet"]) > 0
            assert first_source["score"] > 0  # Positive relevance score
            assert len(first_source["source_metadata"]) > 0

            # Verify the question was preserved
            assert data["question"] == "What is the capital of France?"

            # Verify chunk count is accurate
            assert data["retrieved_chunks_count"] == 1

    def test_acceptance_scenario_2_unanswerable_question_returns_appropriate_response(self):
        """
        Acceptance scenario 2: Unanswerable question returns appropriate response
        """
        # Mock the RAG agent to simulate no content found scenario
        with patch('backend.src.api.endpoints.qa.RAGAgent') as MockRAGAgent:
            # Create a mock instance
            mock_agent_instance = Mock()

            # Mock the answer_question method to return a response indicating no content found
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

            # Assert: Check that the response meets acceptance criteria
            assert response.status_code == 200

            # Parse the response
            data = response.json()

            # Verify that an appropriate response was returned (not an error)
            assert len(data["answer"]) > 0
            assert "cannot find" in data["answer"].lower() or "no information" in data["answer"].lower() or "not found" in data["answer"].lower()

            # Verify that no citations were provided (since no content was found)
            assert len(data["sources"]) == 0
            assert data["retrieved_chunks_count"] == 0

            # Verify the question was preserved
            assert data["question"] == "What is the capital of Mars?"

            # Verify there was no error flag
            assert "error" not in data or data.get("error") is None

    def test_acceptance_scenario_3_grounding_verification(self):
        """
        Additional acceptance scenario: Verify that answers are grounded in retrieved content
        """
        # Mock the RAG agent to simulate a response with clear grounding
        with patch('backend.src.api.endpoints.qa.RAGAgent') as MockRAGAgent:
            # Create a mock instance
            mock_agent_instance = Mock()

            # Mock the answer_question method with content that clearly grounds the answer
            mock_agent_instance.answer_question.return_value = {
                "question": "What is the main topic of the book?",
                "answer": "Based on the retrieved content, the main topic of the book is Physical AI & Humanoid Robotics, as mentioned in the chapter titled 'Introduction to Physical AI'.",
                "sources": [
                    {
                        "content_snippet": "This book covers the fundamentals of Physical AI & Humanoid Robotics...",
                        "score": 0.98,
                        "source_metadata": {
                            "source": "book",
                            "title": "Physical AI & Humanoid Robotics",
                            "chapter": "Introduction",
                            "page": 1
                        }
                    },
                    {
                        "content_snippet": "Physical AI combines artificial intelligence with physical systems...",
                        "score": 0.92,
                        "source_metadata": {
                            "source": "book",
                            "title": "Physical AI & Humanoid Robotics",
                            "chapter": "Foundations",
                            "page": 15
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
                    "question": "What is the main topic of the book?",
                    "top_k": 5
                }
            )

            # Assert: Check that the response demonstrates grounding
            assert response.status_code == 200

            # Parse the response
            data = response.json()

            # Verify the answer references the retrieved content
            assert "retrieved content" in data["answer"].lower() or "based on" in data["answer"].lower()

            # Verify multiple citations are provided
            assert len(data["sources"]) == 2

            # Verify sources have meaningful content and metadata
            for source in data["sources"]:
                assert len(source["content_snippet"]) > 0
                assert source["score"] > 0.5  # Reasonably high relevance
                assert "source" in source["source_metadata"]
                assert "title" in source["source_metadata"]

            # Verify chunk count matches sources
            assert data["retrieved_chunks_count"] == 2

    def test_acceptance_scenario_4_error_handling(self):
        """
        Additional acceptance scenario: Verify proper error handling
        """
        # Mock the RAG agent to simulate an error during processing
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

            # Assert: Check that the error is handled gracefully
            assert response.status_code == 500

            # Parse the response
            error_data = response.json()

            # Verify error response structure
            assert "detail" in error_data
            assert "Error processing question" in error_data["detail"]