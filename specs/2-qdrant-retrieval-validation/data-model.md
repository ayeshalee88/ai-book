# Data Model: Qdrant Retrieval Validation

## Entities

### Query
- **Fields**:
  - `id` (str): Unique identifier for the query
  - `text` (str): The semantic query text
  - `vector` (List[float]): The embedding vector for the query
  - `top_k` (int): Number of results to retrieve (default: 5)
  - `filters` (Dict[str, Any]): Optional filters for the search
  - `created_at` (datetime): Timestamp when query was created

- **Validation Rules**:
  - `text` must not be empty (FR-010)
  - `top_k` must be positive integer
  - `vector` must be a valid embedding vector

### RetrievedChunk
- **Fields**:
  - `id` (str): Unique identifier for the chunk
  - `content` (str): The text content of the retrieved chunk
  - `similarity_score` (float): Similarity score from semantic search (0.0-1.0)
  - `metadata` (Dict[str, Any]): Metadata containing URL, section, chunk index
  - `source_document_id` (str): ID of the original document
  - `position_in_document` (int): Position of the chunk in the original document

- **Validation Rules**:
  - `content` must not be empty
  - `similarity_score` must be between 0.0 and 1.0
  - `metadata` must contain required fields (URL, section, chunk index)

### ValidationLog
- **Fields**:
  - `id` (str): Unique identifier for the log entry
  - `query_id` (str): Reference to the original query
  - `retrieved_chunks` (List[RetrievedChunk]): List of retrieved chunks
  - `relevance_accuracy` (float): Accuracy of retrieved chunks (0.0-1.0)
  - `traceability_accuracy` (float): Accuracy of metadata linking (0.0-1.0)
  - `consistency_score` (float): Consistency across repeated queries (0.0-1.0)
  - `execution_time` (float): Time taken to execute the query (in seconds)
  - `timestamp` (datetime): When the validation was performed
  - `status` (str): Status of the validation (success, partial, failed)

- **Validation Rules**:
  - `relevance_accuracy` must be between 0.0 and 1.0
  - `traceability_accuracy` must be between 0.0 and 1.0
  - `execution_time` must be positive

### ValidationResult
- **Fields**:
  - `id` (str): Unique identifier for the result
  - `query` (Query): The original query that was validated
  - `results` (List[RetrievedChunk]): Retrieved chunks from the validation
  - `metrics` (Dict[str, float]): Various validation metrics
  - `pipeline_ready` (bool): Whether the pipeline is ready for agent usage
  - `issues_found` (List[str]): List of issues identified during validation
  - `created_at` (datetime): When the validation was completed

- **Validation Rules**:
  - `pipeline_ready` is determined by meeting success criteria thresholds
  - `metrics` must include all required success criteria measurements