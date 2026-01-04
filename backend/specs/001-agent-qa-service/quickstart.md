# Quickstart Guide: RAG Chatbot – Agent-Based QA Service

## Prerequisites

- Python 3.11+
- Qdrant vector database running (localhost:6333 by default)
- OpenRouter API key with access to Gemini models
- Book content indexed in Qdrant collection

## Setup

1. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn openai pydantic-settings qdrant-client sentence-transformers
   ```

2. **Set environment variables**:
   ```bash
   export OPENROUTER_API_KEY="your-openrouter-api-key"
   ```

3. **Configure settings** (optional, defaults are set):
   - Qdrant host: localhost
   - Qdrant port: 6333
   - Qdrant collection: book_content
   - OpenRouter model: google/gemini-pro

## Running the Service

1. **Start the API server**:
   ```bash
   cd backend
   python -m src.api.main
   ```

2. **Or with uvicorn directly**:
   ```bash
   cd backend
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Testing the API

1. **Health check**:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

2. **Ask a question**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/qa \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is the main concept discussed in chapter 1?",
       "top_k": 3
     }'
   ```

## API Documentation

- Interactive API docs: http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc

## Configuration

All configuration is managed through environment variables or the `.env` file:

- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `OPENROUTER_MODEL`: Model to use (default: google/gemini-pro)
- `QDRANT_HOST`: Qdrant host (default: localhost)
- `QDRANT_PORT`: Qdrant port (default: 6333)
- `QDRANT_COLLECTION_NAME`: Collection name (default: book_content)
- `APP_HOST`: Application host (default: 0.0.0.0)
- `APP_PORT`: Application port (default: 8000)
- `DEBUG`: Enable debug mode (default: false)