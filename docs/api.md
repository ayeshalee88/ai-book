# API Documentation: Qdrant Retrieval Validation Tool

## Overview

The Qdrant retrieval validation tool provides a modular architecture with clear separation of concerns between data models, services, and the command-line interface.

## Core Models

### Query
Represents a query for Qdrant retrieval validation.

**Attributes:**
- `text`: The query text for semantic search
- `top_k`: Number of results to retrieve (default: 5, range: 1-100)
- `filters`: Optional filters for the query
- `query_vector`: Optional pre-computed query vector

### RetrievedChunk
Represents a chunk retrieved from Qdrant with its metadata.

**Attributes:**
- `id`: Unique identifier for the retrieved chunk
- `text`: The text content of the retrieved chunk
- `score`: Similarity score of the retrieval (0.0-1.0)
- `metadata`: Metadata associated with the chunk
- `position`: Position in the retrieval results
- `collection_name`: Name of the collection where chunk was found
- `timestamp`: When the chunk was retrieved

### ValidationLog
Represents a log entry for a Qdrant retrieval validation operation.

**Attributes:**
- `id`: Unique identifier for the validation log
- `query`: The query that was executed
- `retrieved_chunks`: Chunks retrieved from Qdrant
- `execution_time_ms`: Time taken to execute the query in milliseconds
- `success`: Whether the validation was successful
- `error_message`: Error message if validation failed
- `timestamp`: When the validation was executed
- `metadata`: Additional metadata for the validation

## Services

### QdrantService

Handles all interactions with the Qdrant vector database.

**Methods:**
- `search(query: Query) -> List[RetrievedChunk]`: Execute a semantic search against Qdrant
- `health_check() -> bool`: Check if Qdrant is accessible and healthy
- `get_collection_info() -> Dict[str, Any]`: Get information about the collection

### ValidationService

Provides validation functionality for Qdrant retrieval.

**Methods:**
- `validate_retrieval(query: Query, log_id: str) -> ValidationLog`: Validate retrieval by executing the query
- `validate_top_k_retrieval(query: Query, expected_count: int) -> bool`: Validate correct number of results returned
- `validate_relevance(query: Query, relevance_threshold: float = 0.5) -> bool`: Validate results meet relevance threshold
- `validate_configurable_retrieval(query: Query, log_id: str) -> ValidationLog`: Validate retrieval with configurable parameters
- `validate_with_configurable_top_k(query_text: str, top_k_values: List[int]) -> Dict[int, ValidationLog]`: Validate with different top-k values
- `collect_performance_metrics(validation_logs: List[ValidationLog]) -> PerformanceMetrics`: Collect performance metrics
- `validate_with_performance_tracking(query: Query, log_id: str) -> tuple[ValidationLog, PerformanceMetrics]`: Validate with performance tracking
- `perform_comprehensive_pipeline_validation(validation_queries: List[Query]) -> PipelineReadiness`: Perform comprehensive pipeline validation

## Command-Line Interface

### Available Options

- `--query`: The query text for semantic search (required)
- `--top-k`: Number of results to retrieve (default: 5)
- `--filters`: JSON string of filters to apply to the query
- `--host`: Qdrant host address (default: localhost)
- `--port`: Qdrant port number (default: 6333)
- `--collection`: Name of the Qdrant collection (default: documents)
- `--verbose`: Enable verbose output
- `--performance`: Show performance metrics
- `--multiple-top-k`: Comma-separated list of top-k values to test
- `--pipeline-readiness`: Perform comprehensive pipeline readiness validation
- `--readiness-queries`: Comma-separated list of queries for pipeline validation

### Usage Examples

```python
# Basic usage
from src.cli.validation_cli import main
import sys
import argparse

# You can also use the services directly in your code:
from src.services.qdrant_service import QdrantService
from src.services.validation_service import ValidationService
from src.lib.config import QdrantConfig
from src.models.query import Query

# Initialize services
config = QdrantConfig(host="localhost", port=6333, collection_name="documents")
qdrant_service = QdrantService(config)
validation_service = ValidationService(qdrant_service)

# Create and execute a query
query = Query(text="What is artificial intelligence?", top_k=5)
validation_log = validation_service.validate_retrieval(query, "my_validation_id")

if validation_log.success:
    print(f"Retrieved {len(validation_log.retrieved_chunks)} chunks")
    for chunk in validation_log.retrieved_chunks:
        print(f"- {chunk.text[:100]}... (score: {chunk.score})")
else:
    print(f"Validation failed: {validation_log.error_message}")
```

## Configuration

### Environment Variables

- `QDRANT_HOST`: Qdrant host address
- `QDRANT_PORT`: Qdrant port number
- `QDRANT_API_KEY`: Qdrant API key for authentication
- `QDRANT_COLLECTION_NAME`: Name of the Qdrant collection
- `QDRANT_TIMEOUT`: Request timeout in seconds
- `LOG_LEVEL`: Logging level (default: INFO)

### Programmatic Configuration

```python
from src.lib.config import QdrantConfig

# Create configuration
config = QdrantConfig(
    host="localhost",
    port=6333,
    collection_name="my_collection",
    timeout=60
)

# Or load from environment
config = QdrantConfig.from_env()
```

## Error Handling

The tool provides comprehensive error handling:

- Connection errors are caught and logged
- Query failures are recorded in validation logs
- Performance metrics track success and error rates
- Recommendations are provided when validation fails

## Performance Metrics

The tool tracks several performance metrics:

- Query latency (average, P95, P99)
- Success and error rates
- Throughput (queries per second)
- Memory usage
- Cache hit rates (if applicable)