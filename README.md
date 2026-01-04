# Qdrant Retrieval Validation Tool

A tool for validating Qdrant retrieval functionality that allows AI engineers to execute semantic queries against the Qdrant vector store to validate that embeddings can be reliably retrieved.

## Features

- **Semantic Query Validation**: Execute semantic queries and validate retrieval results
- **Configurable Top-K Retrieval**: Test different numbers of results to retrieve
- **Performance Metrics**: Track latency, success rates, and other performance indicators
- **Pipeline Readiness Validation**: Comprehensive validation of the entire retrieval pipeline
- **Metadata Preservation**: Retrieve and display document metadata with results
- **Error Handling**: Graceful handling of connection and query errors

## Installation

```bash
pip install -e .
```

## Usage

### Basic Validation

```bash
python -m src.cli.validation_cli --query "What is artificial intelligence?"
```

### With Performance Metrics

```bash
python -m src.cli.validation_cli --query "Machine learning concepts" --performance --verbose
```

### Multiple Top-K Validation

```bash
python -m src.cli.validation_cli --query "AI applications" --multiple-top-k "1,3,5,10"
```

### Pipeline Readiness Validation

```bash
python -m src.cli.validation_cli --pipeline-readiness --readiness-queries "query1,query2,query3"
```

### Advanced Options

- `--top-k`: Number of results to retrieve (default: 5)
- `--host`: Qdrant host address (default: localhost)
- `--port`: Qdrant port number (default: 6333)
- `--collection`: Name of the Qdrant collection (default: documents)
- `--filters`: JSON string of filters to apply to the query
- `--verbose`: Enable verbose output
- `--performance`: Show performance metrics

## Configuration

The tool can be configured using environment variables:

- `QDRANT_HOST`: Qdrant host address
- `QDRANT_PORT`: Qdrant port number
- `QDRANT_API_KEY`: Qdrant API key for authentication
- `QDRANT_COLLECTION_NAME`: Name of the Qdrant collection
- `QDRANT_TIMEOUT`: Request timeout in seconds
- `LOG_LEVEL`: Logging level (default: INFO)

## Architecture

The tool follows a modular architecture:

- **Models**: Data models for queries, retrieved chunks, and validation logs
- **Services**: Business logic for Qdrant interaction and validation
- **CLI**: Command-line interface for user interaction
- **Lib**: Utilities for configuration, logging, and authentication

## Development

To run tests:

```bash
pytest tests/
```

To run with development dependencies:

```bash
pip install -e ".[dev]"
```