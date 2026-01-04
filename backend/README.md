# Cohere Embeddings & Vector Storage

This project implements a Python-based backend service that fetches content from deployed URLs, chunks text, generates embeddings using Cohere, and stores them in Qdrant vector database with metadata.

## Features

- Fetches content from the deployed website at https://ai-book-ochre.vercel.app/
- Extracts and cleans HTML content while preserving semantic structure
- Chunks text into 512-token segments with semantic boundaries (paragraphs, headers)
- Generates high-quality embeddings using Cohere's embed-multilingual-v3.0 model
- Stores embeddings in Qdrant with rich metadata (URL, section, chunk index, source type, content hash)
- Supports incremental updates with content hash comparison
- Comprehensive error handling with retry logic and exponential backoff
- Performance tracking and validation metrics
- Protection against SSRF attacks with URL validation

## Prerequisites

- Python 3.8+
- UV package manager
- Cohere API key
- Qdrant Cloud account and API key

## Setup

1. Clone the repository
2. Install dependencies using UV:
   ```bash
   uv sync
   ```

3. Create a `.env` file with the following environment variables:
   ```env
   COHERE_API_KEY=your_cohere_api_key_here
   QDRANT_URL=your_qdrant_url_here
   QDRANT_API_KEY=your_qdrant_api_key_here
   BASE_URL=https://ai-book-ochre.vercel.app/
   ```

## Usage

### Full Ingestion
To process all content from the website:

```bash
python main.py
```

### Incremental Updates
To process only changed content:

```bash
python main.py incremental
```

## Architecture

The system follows a pipeline architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Source URLs   │───▶│  main.py Service │───▶│   Qdrant DB     │
│ (Docusaurus)    │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Cohere API      │
                       │  (Embeddings)    │
                       └──────────────────┘
```

## Components

- `ContentFetcher`: Handles fetching content from URLs with validation and retry logic
- `TextProcessor`: Cleans HTML and chunks text with semantic boundaries
- `CohereEmbedder`: Generates embeddings with retry and rate limiting
- `QdrantStorage`: Stores embeddings with metadata and handles retrieval
- `EmbeddingPipeline`: Orchestrates the entire process

## Security

- URL validation to prevent SSRF attacks
- Rate limiting for API calls
- Environment variable-based configuration for secrets
- Content hash comparison for idempotent operations

## Validation

- 100% page coverage validation
- Retrieval functionality validation
- Performance tracking (should complete within 2 hours)
- Token usage monitoring

## Error Handling

- Retry with exponential backoff for URL fetches
- Cohere API rate limiting with appropriate backoff
- Graceful handling of malformed HTML
- Qdrant unavailability handling
- Content hash comparison for incremental updates