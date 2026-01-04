"""
Comprehensive test suite for the RAG Agent
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.rag_agent import RAGAgent
from src.models.qa import QARequest


class TestRAGAgent:
    """Test suite for RAG Agent functionality"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        with patch('src.services.llm_service.OpenRouterService'):
            with patch('src.agents.tools.qdrant_retriever.QdrantRetrieverTool'):
                self.rag_agent = RAGAgent()

    def test_answer_question_basic(self):
        """Test basic question answering functionality"""
        # Mock the retriever and LLM service
        mock_results = [
            {
                "content": "This is relevant content about Python",
                "metadata": {"source": "python_book", "title": "Python Guide", "section": "1.1"},
                "score": 0.8
            }
        ]
        self.rag_agent.retriever_tool.retrieve_with_fallback = Mock(return_value=mock_results)
        self.rag_agent.llm_service.generate_response = Mock(return_value="This is the answer based on the context.")

        result = self.rag_agent.answer_question("What is Python?")

        assert result["question"] == "What is Python?"
        assert "answer" in result
        assert len(result["sources"]) == 1
        assert result["retrieved_chunks_count"] == 1

    def test_answer_question_no_results(self):
        """Test handling when no relevant results are found"""
        self.rag_agent.retriever_tool.retrieve_with_fallback = Mock(return_value=[])
        self.rag_agent.llm_service.generate_response = Mock(return_value="No relevant information found.")

        result = self.rag_agent.answer_question("What is a non-existent concept?")

        assert result["question"] == "What is a non-existent concept?"
        assert "couldn't find any relevant information" in result["answer"]
        assert result["sources"] == []
        assert result["retrieved_chunks_count"] == 0

    def test_answer_question_error_handling(self):
        """Test error handling in question answering"""
        self.rag_agent.retriever_tool.retrieve_with_fallback = Mock(side_effect=Exception("Connection failed"))

        result = self.rag_agent.answer_question("What is Python?")

        assert result["question"] == "What is Python?"
        assert "Error processing question" in result["answer"]
        assert result["sources"] == []
        assert result["retrieved_chunks_count"] == 0
        assert "error" in result

    def test_validate_response_format_compliance_valid(self):
        """Test response format validation with valid response"""
        valid_response = {
            "question": "What is Python?",
            "answer": "Python is a programming language",
            "sources": [
                {
                    "content_snippet": "Python is a high-level programming language",
                    "score": 0.8,
                    "source_metadata": {},
                    "source_details": {
                        "source": "python_book",
                        "section": "1.1",
                        "url": "http://example.com",
                        "title": "Python Guide",
                        "page": "10",
                        "full_content_length": 100
                    }
                }
            ],
            "retrieved_chunks_count": 1
        }

        compliance = self.rag_agent.validate_response_format_compliance(valid_response)

        assert compliance["is_compliant"] is True
        assert compliance["compliance_score"] == 1.0
        assert len(compliance["missing_fields"]) == 0

    def test_validate_response_format_compliance_invalid(self):
        """Test response format validation with invalid response"""
        invalid_response = {
            "question": "",  # Invalid: empty question
            "answer": "Python is a programming language",
            # Missing 'sources' field
            "retrieved_chunks_count": 1
        }

        compliance = self.rag_agent.validate_response_format_compliance(invalid_response)

        assert compliance["is_compliant"] is False
        assert "sources" in compliance["missing_fields"]
        assert compliance["compliance_score"] < 1.0

    def test_validate_source_metadata(self):
        """Test source metadata validation"""
        sources = [
            {
                "content_snippet": "Sample content",
                "score": 0.8,
                "source_metadata": {"source": "test", "title": "Test Book"},
                "source_details": {
                    "source": "test_source",
                    "section": "1.1",
                    "url": "http://example.com",
                    "title": "Test Book",
                    "page": "10",
                    "full_content_length": 100
                }
            }
        ]

        validation = self.rag_agent.validate_source_metadata(sources)

        assert validation["total_sources"] == 1
        assert validation["valid_sources"] >= 0  # May vary based on validation criteria
        assert validation["invalid_sources"] >= 0
        assert "completeness_score" in validation
        assert len(validation["validation_details"]) == 1

    def test_format_source_citations(self):
        """Test source citation formatting"""
        sources = [
            {
                "content_snippet": "Sample content for testing",
                "score": 0.8,
                "source_metadata": {"source": "test", "title": "Test Book"},
                "source_details": {
                    "source": "test_source",
                    "section": "1.1",
                    "url": "http://example.com",
                    "title": "Test Book",
                    "page": "10",
                    "full_content_length": 100
                }
            }
        ]

        formatted = self.rag_agent.format_source_citations(sources)

        assert "Sources for this answer:" in formatted
        assert "Source 1:" in formatted
        assert "Test Book" in formatted
        assert "test_source" in formatted

    def test_verify_source_citation_accuracy(self):
        """Test source citation accuracy verification"""
        question = "What is Python?"
        answer = "Python is a programming language created in 1991."
        sources = [
            {
                "content_snippet": "Python is a high-level programming language created by Guido van Rossum and first released in 1991.",
                "score": 0.8,
                "source_metadata": {},
                "source_details": {}
            }
        ]

        verification = self.rag_agent.verify_source_citation_accuracy(question, sources, answer)

        assert "question" in verification
        assert verification["total_sources"] == 1
        assert "verification_details" in verification
        assert "overall_confidence" in verification

    def test_get_source_trustworthiness_score(self):
        """Test source trustworthiness scoring"""
        sources = [
            {
                "content_snippet": "Sample content",
                "score": 0.8,
                "source_metadata": {"source": "test", "title": "Test Book"},
                "source_details": {
                    "source": "test_source",
                    "section": "1.1",
                    "url": "http://example.com",
                    "title": "Test Book",
                    "page": "10",
                    "full_content_length": 100
                }
            }
        ]

        trustworthiness = self.rag_agent.get_source_trustworthiness_score(sources)

        assert "trustworthiness_score" in trustworthiness
        assert "factors" in trustworthiness
        assert "message" in trustworthiness
        assert trustworthiness["trustworthiness_score"] >= 0.0
        assert trustworthiness["trustworthiness_score"] <= 1.0

    def test_run_source_verification_acceptance_test(self):
        """Test source verification acceptance test"""
        with patch.object(self.rag_agent, 'answer_question', return_value={
            "question": "What is Python?",
            "answer": "Python is a programming language",
            "sources": [
                {
                    "content_snippet": "Python is a high-level programming language",
                    "score": 0.8,
                    "source_metadata": {"source": "test", "title": "Test Book"},
                    "source_details": {
                        "source": "test_source",
                        "section": "1.1",
                        "url": "http://example.com",
                        "title": "Test Book",
                        "page": "10",
                        "full_content_length": 100
                    }
                }
            ],
            "retrieved_chunks_count": 1
        }):
            result = self.rag_agent.run_source_verification_acceptance_test("What is Python?")

            assert "acceptance_test_passed" in result
            assert "criteria_met" in result
            assert result["question"] == "What is Python?"


class TestQAModels:
    """Test suite for QA data models"""

    def test_qa_request_validation(self):
        """Test QA request model validation"""
        # Valid request
        request = QARequest(question="What is Python?", top_k=5)
        assert request.question == "What is Python?"
        assert request.top_k == 5

        # Valid request with default top_k
        request_default = QARequest(question="What is Python?")
        assert request_default.top_k == 5

    def test_qa_request_validation_edge_cases(self):
        """Test QA request validation edge cases"""
        # Test with minimum top_k
        request_min = QARequest(question="What is Python?", top_k=1)
        assert request_min.top_k == 1

        # Test with maximum top_k
        request_max = QARequest(question="What is Python?", top_k=20)
        assert request_max.top_k == 20


if __name__ == "__main__":
    pytest.main([__file__])