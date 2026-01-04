# Quickstart Guide: Qdrant Retrieval Validation Tool

This guide will help you get started with the Qdrant retrieval validation tool quickly.

## Prerequisites

- Python 3.11 or higher
- Qdrant vector database running and accessible
- Pre-populated collection with embeddings (optional for basic functionality testing)

## Installation

1. Clone or navigate to the project directory
2. Install the package in development mode:

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Basic Usage

### 1. Test Connection

First, test that the tool can connect to your Qdrant instance:

```bash
python -m src.cli.validation_cli --query "test connection" --top-k 1
```

### 2. Perform Basic Validation

Validate retrieval with a simple query:

```bash
python -m src.cli.validation_cli --query "What are the benefits of AI?" --top-k 5
```

### 3. Check Performance

Include performance metrics in your validation:

```bash
python -m src.cli.validation_cli --query "Machine learning" --performance --verbose
```

### 4. Test Multiple Configurations

Test different top-k values:

```bash
python -m src.cli.validation_cli --query "neural networks" --multiple-top-k "1,3,5,10"
```

### 5. Comprehensive Pipeline Validation

Perform a full pipeline readiness check:

```bash
python -m src.cli.validation_cli --pipeline-readiness --readiness-queries "query1,query2,query3"
```

## Configuration

You can configure the tool using environment variables:

```bash
export QDRANT_HOST="your-qdrant-host"
export QDRANT_PORT=6333
export QDRANT_COLLECTION_NAME="your-collection"
export QDRANT_API_KEY="your-api-key"  # if authentication is required
```

Or pass options directly via command line:

```bash
python -m src.cli.validation_cli --query "test" --host your-host --port 6333 --collection your-collection
```

## Common Use Cases

### Validating New Embedding Models

Test how well new embedding models perform with your existing data:

```bash
python -m src.cli.validation_cli --query "sample query for new model" --top-k 10 --performance
```

### Performance Benchmarking

Run multiple queries to benchmark performance:

```bash
python -m src.cli.validation_cli --pipeline-readiness --readiness-queries "query1,query2,query3,query4,query5" --performance
```

### Troubleshooting Retrieval Issues

Use verbose mode to debug retrieval problems:

```bash
python -m src.cli.validation_cli --query "problematic query" --verbose --performance
```

## Next Steps

- Review the full [README](../README.md) for complete documentation
- Run the test suite: `pytest tests/`
- Check out the [API documentation](api.md) for programmatic usage
- Explore advanced configuration options