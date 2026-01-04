"""
Basic test to validate the RAG agent functionality
"""
import os
import sys
from unittest.mock import Mock, patch

# Add the backend directory to the path so we can import the modules
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.agents.rag_agent import RAGAgent
from src.services.llm_service import OpenRouterService
from src.agents.tools.qdrant_retriever import QdrantRetrieverTool


def test_rag_agent_structure():
    """
    Test that the RAG agent can be initialized and has the expected structure
    """
    agent = RAGAgent()

    # Verify agent has required components
    assert hasattr(agent, 'llm_service')
    assert hasattr(agent, 'retriever_tool')
    assert isinstance(agent.llm_service, OpenRouterService)
    assert isinstance(agent.retriever_tool, QdrantRetrieverTool)

    print("✅ RAG Agent structure validation passed")


def test_mocked_qa_flow():
    """
    Test the QA flow with mocked external services
    """
    agent = RAGAgent()

    # Mock the LLM service response
    with patch.object(agent.llm_service, 'generate_response') as mock_llm:
        mock_llm.return_value = "This is a test answer based on the context."

        # Mock the retriever tool response
        with patch.object(agent.retriever_tool, 'get_content_for_agent') as mock_retriever:
            mock_retriever.return_value = "Test context content from Qdrant."

            with patch.object(agent.retriever_tool, 'retrieve') as mock_raw_retrieve:
                mock_raw_retrieve.return_value = [
                    {
                        "content": "Test content",
                        "metadata": {"source": "test_doc", "section": "1.1"},
                        "score": 0.9
                    }
                ]

                # Test the answer_question method
                result = agent.answer_question("What is test?", top_k=1)

                # Verify the result structure
                assert "question" in result
                assert "answer" in result
                assert "sources" in result
                assert "retrieved_chunks_count" in result

                # Verify the content
                assert result["question"] == "What is test?"
                assert result["answer"] == "This is a test answer based on the context."
                assert result["retrieved_chunks_count"] == 1

                print("✅ Mocked QA flow validation passed")


if __name__ == "__main__":
    test_rag_agent_structure()
    test_mocked_qa_flow()
    print("✅ All validations passed! The RAG agent is properly structured and functional.")