"""
Qdrant retriever tool for the RAG agent - Compatible with remote/cloud Qdrant + FastEmbed
"""
from typing import List, Dict, Any
import logging
import json
import re
from src.config.settings import settings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from fastembed import TextEmbedding


def extract_chapter_number(query: str):
    match = re.search(r'chapter\s*(\d+)', query.lower())
    return int(match.group(1)) if match else None

class QdrantRetrieverTool:
    """
    Tool for retrieving relevant content from Qdrant vector database using FastEmbed
    """

    def __init__(self, collection_name: str = "book_content"):
    
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.collection_name = collection_name

        # Lightweight embedding model (same quality as all-MiniLM-L6-v2)
        self.embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        self.logger = logging.getLogger(__name__)

        # Quick validation on init
        if not self.validate_connection():
            self.logger.warning("Qdrant connection failed during initialization")

    def retrieve(self, query: str, top_k: int = 5, min_score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant content chunks for the given query
        """
        try:
            # Generate query embedding with FastEmbed
            query_vector = list(self.embedding_model.embed([query]))[0].tolist()

            # Perform vector search in Qdrant (remote mode)
            search_results = self.client.query_points(
               collection_name=self.collection_name,
               query=query_vector,
               limit=top_k,
               with_payload=True,
               score_threshold=min_score_threshold
            )
            hits = search_results.points

            self.logger.info(f"Raw hits from Qdrant: {len(hits)}")

            results = []
            for point in hits:
                result = ({
                    "content": point.payload.get("content", ""),
                    "metadata": point.payload.get("metadata", {}),
                    "score": point.score
                })
            results.append(result)

            #Fallback: if nothing above threshold, return top results anyway
            if not results and hits:
                self.logger.info(f"No results above threshold {min_score_threshold}, returning top {top_k}")
                for point in hits[:top_k]:
                    results.append({
                        "content": point.payload.get("content", ""),
                        "metadata": point.payload.get("metadata", {}),
                        "score": point.score
                    })
        # Final debug
            self.logger.info(f"Final retrieved chunks: {len(results)}")

            return results

        except Exception as e:
            self.logger.error(f"Error retrieving from Qdrant: {str(e)}")
            return []
               

    def retrieve_with_fallback(self, query: str, top_k: int = 5, min_score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve with fallback strategies when no relevant results are found
        """
        results = self.retrieve(query, top_k, min_score_threshold=0.0)
        if not results:
            self.logger.info(f"No results with threshold {min_score_threshold}, trying 0.05")
            results = self.retrieve(query, top_k, 0.05)
        if not results:
            self.logger.info(f"No results, broadening search to top_k * 2")
            results = self.retrieve(query, top_k * 2, 0.01)
        return results

    def validate_connection(self) -> bool:
        """Check if we can connect to Qdrant"""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            self.logger.error(f"Qdrant connection validation failed: {str(e)}")
            return False

    def validate_collection_exists(self) -> bool:
        """Check if the target collection exists"""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            self.logger.error(f"Collection '{self.collection_name}' does not exist")
            return False

    def get_content_for_agent(self, query: str, top_k: int = 5) -> str:
        """Get formatted content for the agent with grounding info"""
        results = self.retrieve(query, top_k)
        if not results:
            return "No relevant content found in the knowledge base."

        formatted = "Retrieved content for answering the question:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"Result {i}:\n"
            formatted += f"Content: {result['content']}\n"
            formatted += f"Metadata: {json.dumps(result['metadata'])}\n\n"

        return formatted

    def get_detailed_source_info(self, query: str, top_k: int = 5) -> list:
        """Get detailed source information with enhanced metadata"""
        results = self.retrieve(query, top_k)
        detailed_sources = []

        for result in results:
            metadata = result.get('metadata', {})
            additional = {k: v for k, v in metadata.items()
                          if k not in ['source', 'section', 'url', 'title', 'page']}

            source_info = {
                "content_snippet": result.get('content', '')[:200] + "..." 
                                   if len(result.get('content', '')) > 200 
                                   else result.get('content', ''),
                "score": result.get('score', 0),
                "source_metadata": metadata,
                "source_details": {
                    "source": metadata.get('source', 'Unknown'),
                    "section": metadata.get('section', 'Unknown'),
                    "url": metadata.get('url', 'Unknown'),
                    "title": metadata.get('title', 'Unknown'),
                    "page": metadata.get('page', 'Unknown'),
                    "full_content_length": len(result.get('content', '')),
                    "additional_metadata": additional
                }
            }
            detailed_sources.append(source_info)

        return detailed_sources