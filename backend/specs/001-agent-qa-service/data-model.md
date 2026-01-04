# Data Model: RAG Chatbot – Agent-Based QA Service

## Overview

This document defines the data structures and models used in the RAG Chatbot service. The service primarily works with question-answer interactions and maintains clear separation between user queries, retrieved content, and generated responses.

## Core Data Models

### 1. Request Models

#### QARequest
```python
{
  "question": "str",        # The user's natural language question
  "top_k": "int"            # Number of relevant chunks to retrieve (optional, default: 5)
}
```

**Validation Rules:**
- Question must be non-empty string
- Question length should be reasonable (e.g., < 1000 characters)
- top_k should be between 1 and 20

### 2. Response Models

#### QAResponse
```python
{
  "question": "str",                    # Echo of the original question
  "answer": "str",                      # The generated answer from the LLM
  "sources": [                          # Array of source information
    {
      "content_snippet": "str",         # First 200 chars of the retrieved content
      "score": "float",                 # Similarity score from Qdrant
      "source_metadata": {              # Metadata from the original content
        # Any metadata fields from Qdrant payload
      }
    }
  ],
  "retrieved_chunks_count": "int",      # Number of chunks actually retrieved
  "error": "str"                        # Error message if processing failed (optional)
}
```

#### HealthCheckResponse
```python
{
  "status": "str",                      # "healthy" or "unhealthy"
  "services": {                         # Status of individual services
    "llm_service": "bool",
    "retriever_service": "bool",
    "overall": "bool"
  }
}
```

## Internal Data Structures

### 1. Agent Internal State

#### AgentResult
```python
{
  "question": "str",
  "answer": "str",
  "sources": "list[dict]",              # Same format as sources in QAResponse
  "retrieved_chunks_count": "int",
  "error": "str"                        # Optional error field
}
```

### 2. Retrieval Data Structure

#### RetrievedChunk
```python
{
  "content": "str",                     # Full content retrieved from Qdrant
  "metadata": {                         # Metadata associated with the content
    # Any metadata fields stored in Qdrant
  },
  "score": "float"                      # Similarity score from Qdrant
}
```

## Qdrant Collection Schema

The service expects a Qdrant collection with the following structure:

### Collection: `book_content`
- **Vector size**: Depends on the embedding model used (e.g., 384 for all-MiniLM-L6-v2)
- **Distance metric**: Cosine similarity

### Payload Structure:
```json
{
  "content": "str",                     // The text content
  "metadata": {
    "source": "str",                    // Source document identifier
    "section": "str",                   // Section/chapter identifier
    "url": "str",                       // URL or path to original content
    "title": "str",                     // Title of the content chunk
    "page": "int",                      // Page number if applicable
    // Any other relevant metadata
  }
}
```

## API Message Format

### OpenRouter Communication

The service constructs messages in the following format for OpenRouter API:

```python
[
  {
    "role": "system",
    "content": "You are a helpful assistant that answers questions based only on the provided context. Do not make up information. If the context doesn't contain enough information to answer, say so."
  },
  {
    "role": "system",
    "content": "Context for answering the question:\n[retrieved content chunks]"
  },
  {
    "role": "user",
    "content": "[user's question]"
  }
]
```

## Data Flow

1. **Input**: User provides a question via QARequest
2. **Retrieval**: System retrieves relevant content from Qdrant
3. **Processing**: Retrieved content is formatted and sent to LLM
4. **Generation**: LLM generates an answer based on the context
5. **Output**: System returns QAResponse with answer and source information

## Validation Rules

### Content Validation
- All content must be UTF-8 encoded text
- Maximum content length for responses is 1000 tokens (approximately)
- Source metadata must be valid JSON-serializable objects

### Error Handling
- If retrieval fails, return appropriate error in response
- If LLM call fails, return error message to user
- If no relevant content is found, indicate this in the response

## Performance Considerations

### Size Limits
- Individual content chunks should be limited to reasonable size (e.g., < 1000 tokens each)
- Total context sent to LLM should respect token limits
- Response size should be optimized for network transmission

### Caching Opportunities
- Frequently retrieved content could be cached
- Embedding results could be cached for repeated queries
- Response caching for identical questions

## Extensibility

### Future Data Additions
- User session information for conversation context
- Feedback data for continuous improvement
- Usage analytics (while respecting privacy)
- Response confidence scores