"""
QA API endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from ...agents.rag_agent import RAGAgent
from ...models.qa import QARequest, QAResponse, HealthCheckResponse
from ...config.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize the RAG agent
rag_agent = RAGAgent()


@router.post("/qa", response_model=QAResponse)
async def ask_question(request: QARequest) -> QAResponse:
    """
    Endpoint to ask a question and get a grounded answer
    """
    import time
    from pydantic import ValidationError
    import traceback

    start_time = time.time()

    try:
        # Validate input parameters
        if not request.question or not request.question.strip():
            logger.warning("QA request received with empty question")
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Validate question length
        if len(request.question) > 1000:  # Maximum 1000 characters
            logger.warning(f"QA request received with overly long question ({len(request.question)} chars)")
            raise HTTPException(status_code=400, detail="Question is too long. Maximum 1000 characters allowed.")

        # Validate question format - check for potential injection attempts or malformed content
        question_stripped = request.question.strip()

        # Check for excessive special characters that might indicate injection attempts
        special_char_ratio = sum(1 for c in question_stripped if not c.isalnum() and not c.isspace()) / len(question_stripped) if question_stripped else 0
        if special_char_ratio > 0.5:  # More than 50% special characters
            logger.warning(f"QA request received with high ratio of special characters ({special_char_ratio:.2f})")
            raise HTTPException(status_code=400, detail="Question contains too many special characters. Please rephrase.")

        # Check for potential script tags or other HTML-like content
        question_lower = question_stripped.lower()
        if any(tag in question_lower for tag in ['<script', 'javascript:', 'onerror=', 'onload=']):
            logger.warning(f"QA request received with potential injection content: {question_stripped[:50]}...")
            raise HTTPException(status_code=400, detail="Question contains invalid content. Please remove any script tags or special characters.")

        if request.top_k < 1 or request.top_k > 20:
            logger.warning(f"QA request received with invalid top_k: {request.top_k}")
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

        # Log incoming request
        logger.info(f"QA request received - Question: {request.question[:50]}..., Top_k: {request.top_k}")

        # Get the answer from the RAG agent
        result = rag_agent.answer_question(
            question=request.question,
            top_k=request.top_k
        )

        # Check if the result contains an error from the agent
        if result.get('error'):
            logger.warning(f"RAG agent returned error: {result.get('error')}")
            raise HTTPException(status_code=502, detail=f"Service error: {result.get('error')}")

        # Calculate response time
        response_time = time.time() - start_time

        # Log successful response
        logger.info(f"QA request completed - Response time: {response_time:.2f}s, Sources found: {result.get('retrieved_chunks_count', 0)}")

        # Log response summary (without full answer to avoid log bloat)
        logger.debug(f"Response summary - Question length: {len(request.question)}, Answer length: {len(result.get('answer', ''))}, Sources: {len(result.get('sources', []))}")

        return QAResponse(**result)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValidationError as e:
        response_time = time.time() - start_time
        logger.error(f"Validation error for QA request - Question: {request.question[:50] if request.question else 'None'}, Error: {str(e)}, Response time: {response_time:.2f}s")
        raise HTTPException(status_code=422, detail=f"Invalid request format: {str(e)}")
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"Unexpected error in QA request - Question: {request.question[:50] if request.question else 'None'}, Error: {str(e)}, Traceback: {traceback.format_exc()}, Response time: {response_time:.2f}s")

        # Return a more informative error response for integration scenarios
        error_detail = {
            "error": "Internal server error",
            "message": "An error occurred while processing your request",
            "request_id": f"qa-{int(start_time)}",  # Simple request ID for tracking
            "timestamp": time.time(),
            "status_code": 500
        }

        raise HTTPException(
            status_code=500,
            detail=error_detail
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint to verify all services are available
    """
    try:
        validation_result = rag_agent.validate_setup()

        status = "healthy" if validation_result.get("overall", False) else "unhealthy"

        return HealthCheckResponse(
            status=status,
            services=validation_result
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            services={
                "llm_service": False,
                "retriever_service": False,
                "overall": False
            }
        )


@router.get("/validate", response_model=Dict[str, Any])
async def validate_setup() -> Dict[str, Any]:
    """
    Validate that all required services are properly configured
    """
    try:
        validation_result = rag_agent.validate_setup()
        return {
            "validation_result": validation_result,
            "message": "Setup validation completed",
            "openrouter_model": settings.openrouter_model,
            "qdrant_collection": settings.qdrant_collection_name
        }
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")