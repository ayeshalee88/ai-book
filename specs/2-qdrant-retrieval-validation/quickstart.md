# Quickstart: Qdrant Retrieval Validation

## Prerequisites
- Python 3.11+
- Qdrant vector database instance running and accessible
- Pre-generated embeddings stored in Qdrant

## Installation
```bash
pip install qdrant-client python-dotenv
```

## Configuration
Create a `.env` file with the following:
```env
QDRANT_HOST=your-qdrant-host
QDRANT_PORT=6333
QDRANT_API_KEY=your-api-key-if-required
VALIDATION_TOKEN=your-validation-auth-token
```

## Basic Usage
```bash
# Validate a single query
python -m src.cli.validation_cli --query "What is the main concept?" --top-k 5

# Run batch validation with multiple queries
python -m src.cli.validation_cli --batch queries.txt

# Validate pipeline readiness
python -m src.cli.validation_cli --validate-readiness
```

## Programmatic Usage
```python
from src.services.validation_service import ValidationService

# Initialize the validation service
validator = ValidationService()

# Run a validation query
result = validator.validate_query(
    query_text="What is the main concept?",
    top_k=5
)

# Check if pipeline is ready
if result.validation_metrics['pipeline_ready']:
    print("Pipeline is ready for agent usage!")
else:
    print(f"Pipeline needs improvements: {result.issues_found}")
```

## Validation Output
The validation will produce:
- Retrieved text chunks ranked by similarity
- Retrieval quality metrics (relevance, traceability, consistency)
- Diagnostic logs for each query
- Confirmation of pipeline readiness for agent usage