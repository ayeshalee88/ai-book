# Tasks: Cohere Embeddings & Vector Storage

## Feature Overview
This feature implements a Python-based backend service that fetches content from deployed URLs, chunks text, generates embeddings using Cohere, and stores them in Qdrant vector database with metadata. The implementation will be contained in a single main.py file using the UV package manager for dependency management.

**Target Site**: https://ai-book-ochre.vercel.app/
**Sitemap URL**: https://ai-book-ochre.vercel.app/sitemap.xml

## Phase 1: Setup
Initialize project with UV package manager and required dependencies.

- [X] T001 Create project directory structure
- [X] T002 Initialize Python project with UV package manager
- [X] T003 [P] Install dependencies: requests, cohere, qdrant-client, beautifulsoup4, tenacity
- [X] T004 Create .env file template with required environment variables
- [X] T005 Create requirements.txt file for dependency tracking

## Phase 2: Foundational Components
Core components needed for all user stories.

- [X] T006 Create ContentChunk dataclass with text, url, section, chunk_index, source_type, content_hash
- [X] T007 Implement CohereEmbedder class with generate_embeddings method
- [X] T008 Implement QdrantStorage class with collection creation and upsert functionality
- [X] T009 Implement ContentFetcher class with URL fetching and extraction functionality
- [X] T010 Implement TextProcessor class with clean_content and chunk_text methods
- [X] T011 Create EmbeddingPipeline class to orchestrate the process

## Phase 3: User Story 1 - Content Embedding Process (P1)
As an AI engineer, I want to automatically extract content from the deployed book website and generate vector embeddings so that I can enable semantic search capabilities for the RAG chatbot.

**Independent Test**: The system can extract content from the entire website, generate embeddings using Cohere, and store them in Qdrant with complete metadata.

- [X] T012 [US1] Implement get_all_urls function to extract all URLs from base site
- [X] T013 [US1] Implement extract_text_from_url function to extract clean text from a URL
- [X] T014 [US1] Implement embed function to generate embeddings using Cohere API
- [X] T015 [US1] Implement create_collection function to create Qdrant collection named "ragembeddings"
- [X] T016 [US1] Implement save_chunk_to_qdrant function to save content with embeddings to Qdrant
- [X] T017 [US1] Implement ingest_book function to orchestrate the entire ingestion process
- [X] T018 [US1] Implement main function to execute the pipeline with the target site
- [X] T019 [US1] Add proper error handling and logging to all ingestion components

## Phase 4: User Story 2 - Content Chunking and Metadata Storage (P2)
As a developer, I want the system to properly chunk content and store relevant metadata so that I can trace embeddings back to their original source and maintain context during retrieval.

**Independent Test**: Content is split into appropriately sized chunks with metadata including page URL, section, chunk index, and source type, allowing for precise source attribution during retrieval.

- [X] T020 [US2] Enhance chunk_text function to use 512-token segments with semantic boundaries (paragraphs, headers)
- [X] T021 [US2] Ensure metadata includes: page URL, section, chunk index, and source type with each embedding
- [X] T022 [US2] Implement content hash generation for each chunk to enable traceability
- [X] T023 [US2] Validate that each embedding in Qdrant can be traced back to its original source
- [X] T024 [US2] Add logging to track chunk creation and metadata assignment

## Phase 5: User Story 3 - Incremental Updates (P3)
As a system maintainer, I want the embedding process to support incremental updates so that I can efficiently update the vector store when new content is added or existing content changes.

**Independent Test**: When new or updated content exists, only those specific pages/chunks are processed, saving computational resources and time.

- [X] T025 [US3] Implement content hash comparison to detect content changes
- [X] T026 [US3] Add idempotent operations to prevent duplicate embeddings
- [X] T027 [US3] Implement logic to process only changed content during incremental updates
- [X] T028 [US3] Add functionality to identify and re-embed updated content
- [X] T029 [US3] Create mechanism to identify new content for embedding

## Phase 6: Error Handling & Resilience
Implement robust error handling and resilience features.

- [X] T030 Implement retry with exponential backoff for URL fetches
- [X] T031 [P] Implement retry with exponential backoff for Cohere API calls
- [X] T032 Handle Cohere rate limiting with appropriate backoff and retry logic
- [X] T033 Handle malformed HTML gracefully with fallback parsing strategies
- [X] T034 Handle very large pages by splitting into overlapping segments with context bridges
- [X] T035 Implement graceful handling of Qdrant unavailability

## Phase 7: Validation & Logging
Implement validation and comprehensive logging.

- [X] T036 Implement validate_retrieval function to test retrieval functionality
- [X] T037 Add comprehensive logging throughout the ingestion pipeline
- [X] T038 Create validation checks for 100% page coverage
- [X] T039 Implement performance tracking to ensure completion within 2 hours
- [X] T040 Add metrics for monitoring API usage and costs

## Phase 8: Polish & Cross-Cutting Concerns
Final touches and cross-cutting concerns.

- [X] T041 Add URL validation to prevent SSRF attacks
- [X] T042 Implement rate limiting to prevent API abuse
- [X] T043 Add data privacy checks to ensure no sensitive content is processed
- [X] T044 Implement batch processing for optimal API call efficiency
- [X] T045 Add connection pooling for HTTP requests and Qdrant connections
- [X] T046 Optimize memory management for large documents
- [X] T047 Create comprehensive README with setup and execution instructions

## Dependencies
- User Story 1 (P1) must be completed before User Story 2 (P2) and User Story 3 (P3)
- Foundational components (Phase 2) must be completed before any user story phases
- Setup phase (Phase 1) must be completed before any other phases

## Parallel Execution Examples
- T003 [P], T004, T005 can be executed in parallel during setup
- T007, T008, T009, T010 can be developed in parallel during foundational phase
- T030, T031, T032, T033 can be implemented in parallel during error handling phase

## Implementation Strategy
- MVP: Complete Phase 1, Phase 2, and Phase 3 (User Story 1) to get basic functionality
- Increment 1: Add User Story 2 (content chunking and metadata)
- Increment 2: Add User Story 3 (incremental updates)
- Increment 3: Add error handling and validation
- Final: Polish and cross-cutting concerns