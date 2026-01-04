"""Qdrant service for retrieval validation."""

from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.models.query import Query
from src.models.retrieved_chunk import RetrievedChunk
from src.lib.config import QdrantConfig


class QdrantService:
    """Service for interacting with Qdrant vector database."""

    def __init__(self, config: QdrantConfig):
        """Initialize the Qdrant service with configuration."""
        self.config = config
        self.client = QdrantClient(
            host=config.host,
            port=config.port,
            api_key=config.api_key,
            timeout=config.timeout
        )

    def search(self, query: Query) -> List[RetrievedChunk]:
        """Execute a semantic search against Qdrant and return retrieved chunks."""
        # Prepare the search request
        search_params = models.SearchParams(hnsw_ef=128)

        # Convert query filters to Qdrant format if provided
        qdrant_filters = None
        if query.filters:
            qdrant_filters = self._convert_filters_to_qdrant_format(query.filters)

        # Perform the search
        results = self.client.search(
            collection_name=self.config.collection_name,
            query_text=query.text if not query.query_vector else None,
            query_vector=query.query_vector,
            limit=query.top_k,
            filter=qdrant_filters,
            params=search_params
        )

        # Convert results to RetrievedChunk objects
        retrieved_chunks = []
        for idx, result in enumerate(results):
            # Extract text from payload
            text = result.payload.get("text", "")

            # Extract all other metadata except 'text'
            metadata = {k: v for k, v in result.payload.items() if k != "text"}

            chunk = RetrievedChunk(
                id=str(result.id),
                text=text,
                score=result.score,
                metadata=metadata,
                position=idx,
                collection_name=self.config.collection_name
            )
            retrieved_chunks.append(chunk)

        return retrieved_chunks

    def _convert_filters_to_qdrant_format(self, filters: Dict[str, Any]) -> models.Filter:
        """Convert simple filters to Qdrant Filter format."""
        conditions = []
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            elif isinstance(value, list):
                # Handle "in" queries
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value)
                    )
                )

        return models.Filter(must=conditions) if conditions else None

    def health_check(self) -> bool:
        """Check if Qdrant is accessible and healthy."""
        try:
            # Try to get collection info as a basic health check
            self.client.get_collection(self.config.collection_name)
            return True
        except Exception:
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            collection_info = self.client.get_collection(self.config.collection_name)
            return {
                "name": collection_info.config.params.vectors.size,
                "vector_size": collection_info.config.params.vectors.size,
                "points_count": collection_info.points_count,
                "config": collection_info.config.dict()
            }
        except Exception as e:
            return {"error": str(e)}