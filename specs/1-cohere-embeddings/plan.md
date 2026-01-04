# Implementation Plan: Cohere Embeddings & Vector Storage

## Summary
This plan outlines the architecture for a Python-based backend service that fetches content from deployed URLs, chunks text, generates embeddings using Cohere, and stores them in Qdrant vector database with metadata. The implementation will be contained in a single main.py file using the UV package manager for dependency management.

## Architecture
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

The system follows a pipeline architecture where content is extracted from URLs, processed through cleaning and chunking, embedded using Cohere, and stored in Qdrant with appropriate metadata.

## Technical Approach
1. **Project Setup**: Initialize project with UV package manager and install required dependencies (requests, cohere, qdrant-client, beautifulsoup4)
2. **URL Fetching**: Implement robust HTTP client to fetch content from deployed URLs with proper error handling and retries
3. **Content Processing**: Parse HTML content, extract text while preserving structure, and implement intelligent chunking based on semantic boundaries
4. **Embedding Generation**: Use Cohere API to generate vector embeddings for text chunks with proper rate limiting
5. **Vector Storage**: Store embeddings in Qdrant with metadata (URL, section, chunk index, source type) and implement upsert logic
6. **Validation**: Implement retrieval validation and logging mechanisms
7. **Target Site**: https://ai-book-ochre.vercel.app/
8. **Sitemap URL**: https://ai-book-ochre.vercel.app/sitemap.xml

## Key Decisions

### 1. Single File Architecture
- **Decision**: Implement entire functionality in a single main.py file
- **Rationale**: Simplifies deployment and maintenance for this focused feature
- **Impact**: All components (fetching, processing, embedding, storage) in one file

### 2. UV Package Management
- **Decision**: Use UV package manager instead of pip
- **Rationale**: Faster dependency resolution and better performance than traditional pip
- **Impact**: Requires uv installation and different command syntax for dependency management

### 3. Cohere Embedding Model Selection
- **Decision**: Use Cohere's embed-multilingual-v3.0 model
- **Rationale**: Best performance for diverse content types and supports multiple languages
- **Impact**: Higher quality embeddings but requires Cohere API key

### 4. Qdrant Collection Strategy
- **Decision**: Use a single collection with metadata filtering
- **Rationale**: Simplifies management and allows efficient querying across all content
- **Impact**: All embeddings stored in one collection with rich metadata for filtering

📋 Architectural decision detected: Embedding model selection and collection strategy — Document reasoning and tradeoffs? Run `/sp.adr embedding-architecture`

## Data Flow
1. **Input**: Deployed website URL(s) from configuration
2. **Fetch**: HTTP requests to retrieve HTML content from URLs
3. **Parse**: Extract text content while preserving semantic structure
4. **Chunk**: Split content into 512-token segments using semantic boundaries
5. **Embed**: Generate vector embeddings using Cohere API
6. **Store**: Upsert embeddings into Qdrant with metadata
7. **Validate**: Test retrieval functionality and log results

## Error Handling & Resilience
- **HTTP Failures**: Implement retry with exponential backoff for failed URL fetches
- **API Rate Limits**: Handle Cohere rate limiting with exponential backoff and retry logic
- **Qdrant Unavailability**: Queue failed upserts for later processing when Qdrant is unavailable
- **Content Extraction**: Gracefully handle malformed HTML with fallback parsing strategies
- **Token Limits**: Split oversized content into manageable chunks before embedding

## Security Considerations
- **API Keys**: Store Cohere and Qdrant credentials securely using environment variables
- **URL Validation**: Validate input URLs to prevent SSRF attacks
- **Rate Limiting**: Implement proper rate limiting to prevent API abuse
- **Data Privacy**: Ensure no sensitive content is inadvertently processed or stored

## Performance & Scalability
- **Batch Processing**: Process multiple chunks in batches to optimize API calls
- **Caching**: Cache successful URL fetches to avoid redundant requests
- **Connection Pooling**: Use connection pooling for HTTP requests and Qdrant connections
- **Memory Management**: Process large documents in chunks to avoid memory issues

## Implementation Phases
1. **Phase 1**: Basic project setup with UV and core dependencies
2. **Phase 2**: URL fetching and content extraction functionality
3. **Phase 3**: Text processing and chunking implementation
4. **Phase 4**: Cohere integration and embedding generation
5. **Phase 5**: Qdrant storage and metadata handling
6. **Phase 6**: Validation and logging implementation
7. **Phase 7**: Error handling and resilience features

## Risks & Mitigations
- **API Costs**: Cohere usage may incur costs; implement usage tracking and limits
- **Rate Limiting**: API rate limits could slow processing; implement smart queuing
- **Large Content**: Very large documents could exceed memory limits; implement streaming processing
- **Qdrant Limits**: Free tier limitations may impact functionality; design for scalability