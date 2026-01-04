"""
Main FastAPI application for the RAG Chatbot service
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from .endpoints.qa import router as qa_router
from ..config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO if not settings.debug else logging.DEBUG)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot - Agent-Based QA Service",
    description="""
    An agent-based question answering service that uses Qdrant retrieval and Gemini models via OpenRouter.

    ## Features
    - Ask questions about technical book content
    - Receive grounded answers with source citations
    - Verify source information and trustworthiness
    - Health checks and service validation

    ## Usage
    Send POST requests to `/api/v1/qa` with your question in the request body.
    """,
    version="1.0.0",
    contact={
        "name": "AI Book RAG Service",
        "url": "http://localhost:8000",
        "email": "support@example.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(qa_router, prefix="/api/v1", tags=["qa"])

@app.get("/")
async def root():
    """
    Root endpoint for basic service information
    """
    return {
        "message": "RAG Chatbot - Agent-Based QA Service",
        "version": "1.0.0",
        "endpoints": [
            {
                "path": "/api/v1/qa",
                "method": "POST",
                "description": "Ask a question and get a grounded answer"
            },
            {
                "path": "/api/v1/health",
                "method": "GET",
                "description": "Health check for the service"
            },
            {
                "path": "/api/v1/validate",
                "method": "GET",
                "description": "Validate service setup"
            }
        ]
    }

@app.on_event('startup')
async def startup_event():
    """
    Perform startup validation
    """
    logger.info("Starting up RAG Chatbot service...")

    # Perform basic validation of services
    try:
        from ..agents.rag_agent import RAGAgent
        agent = RAGAgent()
        validation_result = agent.validate_setup()
        logger.info(f"Service validation result: {validation_result}")

        if not validation_result.get("overall", False):
            logger.warning("Some services failed validation, check configuration")
        else:
            logger.info("All services validated successfully")
    except Exception as e:
        logger.error(f"Error during startup validation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug
    )

app=app  # For easy import elsewhere