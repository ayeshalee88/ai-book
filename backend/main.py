import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ContentChunk:
    """Represents a chunk of content with metadata"""
    text: str
    url: str
    section: str
    chunk_index: int
    source_type: str
    content_hash: str


class CohereEmbedder:
    """Handles embedding generation using Cohere API"""

    def __init__(self, api_key: str):
        self.client = cohere.Client(api_key)
        self.model = "embed-multilingual-v3.0"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with retry logic"""
        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="search_document"
        )
        return response.embeddings


class QdrantStorage:
    """Handles storage and retrieval of embeddings in Qdrant"""

    def __init__(self, url: str, api_key: str, collection_name: str = "ragembeddings"):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensure the collection exists with proper configuration"""
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} already exists")
        except:
            # Create collection with appropriate vector size (Cohere embeddings are 1024-dim)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
            )
            logger.info(f"Created collection {self.collection_name}")

    def upsert_embeddings(self, chunks: List[ContentChunk], embeddings: List[List[float]]):
        """Upsert embeddings with metadata into Qdrant"""
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = models.PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "url": chunk.url,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "source_type": chunk.source_type,
                    "content_hash": chunk.content_hash,
                    "text": chunk.text
                }
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Upserted {len(points)} embeddings into Qdrant")

    def validate_retrieval(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Validate retrieval functionality"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        return [
            {
                "url": result.payload.get("url"),
                "text": result.payload.get("text"),
                "score": result.score
            }
            for result in results
        ]


class ContentFetcher:
    """Handles fetching content from URLs"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def fetch_content(self, url: str) -> Optional[str]:
        """Fetch content from URL with retry logic and URL validation"""
        # Validate URL to prevent SSRF attacks
        if not self._is_valid_url(url):
            logger.error(f"Invalid URL detected (potential SSRF): {url}")
            return None

        response = self.session.get(url)
        response.raise_for_status()
        return response.text

    def extract_all_urls(self) -> List[str]:
        """Extract all URLs from the base site"""
        content = self.fetch_content(self.base_url)
        if not content:
            return []

        soup = BeautifulSoup(content, 'html.parser')
        links = soup.find_all('a', href=True)

        urls = set()
        for link in links:
            href = link['href']
            full_url = urljoin(self.base_url, href)

            # Validate URL before adding to prevent SSRF
            if self.base_url in full_url and self._is_valid_url(full_url) and full_url not in urls:
                urls.add(full_url)

        # Also add the base URL itself if it's valid
        if self._is_valid_url(self.base_url):
            urls.add(self.base_url)
        return list(urls)

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL to prevent SSRF attacks"""
        try:
            parsed = urlparse(url)

            # Check if it's a valid HTTP/HTTPS URL
            if parsed.scheme not in ['http', 'https']:
                return False

            # Check for private IP addresses (potential SSRF protection)
            hostname = parsed.hostname
            if hostname:
                # Block private IP ranges
                if hostname in ['localhost', '127.0.0.1', '::1']:
                    return False
                # Check for internal IP patterns (simplified)
                if hostname.startswith('10.') or hostname.startswith('172.') or hostname.startswith('192.168.'):
                    return False

            # Ensure URL is within the allowed domain
            base_parsed = urlparse(self.base_url)
            if parsed.netloc != base_parsed.netloc:
                # Allow subdomains but not completely different domains
                if not parsed.netloc.endswith(base_parsed.netloc):
                    return False

            return True
        except Exception:
            return False


class TextProcessor:
    """Handles text cleaning and chunking"""

    def __init__(self, max_chunk_size: int = 512):
        self.max_chunk_size = max_chunk_size

    def clean_content(self, html_content: str) -> str:
        """Clean HTML content to extract text"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text content
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    def chunk_text(self, text: str, url: str) -> List[ContentChunk]:
        """Chunk text into 512-token segments with semantic boundaries (paragraphs, headers)"""
        # Split text by paragraphs first to maintain semantic boundaries
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would exceed the limit, finalize current chunk
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                # Create a chunk with metadata
                content_hash = hashlib.md5(current_chunk.encode()).hexdigest()
                chunk = ContentChunk(
                    text=current_chunk.strip(),
                    url=url,
                    section=url.split('/')[-1] or 'index',
                    chunk_index=chunk_index,
                    source_type='text',
                    content_hash=content_hash
                )
                chunks.append(chunk)
                logger.info(f"Created chunk {chunk_index} with {len(current_chunk)} chars from {url}")

                # Start new chunk with this paragraph
                current_chunk = paragraph
                chunk_index += 1
            elif len(paragraph) <= self.max_chunk_size:
                # Add paragraph to current chunk if it fits
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Paragraph is too long, split it into smaller pieces
                # Handle very large pages by splitting into overlapping segments with context bridges
                sub_chunks = self._split_large_paragraph(paragraph, url, chunk_index)
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)

        # Add the last chunk if it has content
        if current_chunk.strip():
            content_hash = hashlib.md5(current_chunk.encode()).hexdigest()
            chunk = ContentChunk(
                text=current_chunk.strip(),
                url=url,
                section=url.split('/')[-1] or 'index',
                chunk_index=chunk_index,
                source_type='text',
                content_hash=content_hash
            )
            chunks.append(chunk)
            logger.info(f"Created final chunk {chunk_index} with {len(current_chunk)} chars from {url}")

        return chunks

    def _split_large_paragraph(self, paragraph: str, url: str, start_chunk_index: int) -> List[ContentChunk]:
        """Split a large paragraph into smaller chunks while preserving sentence boundaries"""
        sentences = paragraph.split('. ')
        sub_chunks = []
        current_sub_chunk = ""
        chunk_index = start_chunk_index

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence would exceed the limit, finalize current sub-chunk
            if len(current_sub_chunk) + len(sentence) > self.max_chunk_size and current_sub_chunk:
                content_hash = hashlib.md5(current_sub_chunk.encode()).hexdigest()
                chunk = ContentChunk(
                    text=current_sub_chunk.strip() + ".",
                    url=url,
                    section=url.split('/')[-1] or 'index',
                    chunk_index=chunk_index,
                    source_type='text',
                    content_hash=content_hash
                )
                sub_chunks.append(chunk)
                logger.info(f"Created sub-chunk {chunk_index} with {len(current_sub_chunk)} chars from {url}")

                current_sub_chunk = sentence + ". "
                chunk_index += 1
            else:
                current_sub_chunk += sentence + ". "

        # Add the last sub-chunk if it has content
        if current_sub_chunk.strip():
            content_hash = hashlib.md5(current_sub_chunk.encode()).hexdigest()
            chunk = ContentChunk(
                text=current_sub_chunk.strip(),
                url=url,
                section=url.split('/')[-1] or 'index',
                chunk_index=chunk_index,
                source_type='text',
                content_hash=content_hash
            )
            sub_chunks.append(chunk)
            logger.info(f"Created final sub-chunk {chunk_index} with {len(current_sub_chunk)} chars from {url}")

        return sub_chunks


class EmbeddingPipeline:
    """Main pipeline for the embedding process"""

    def __init__(self):
        # Initialize components
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.base_url = os.getenv("BASE_URL", "https://ai-book-ochre.vercel.app/")

        if not all([self.cohere_api_key, self.qdrant_url, self.qdrant_api_key]):
            raise ValueError("Missing required environment variables: COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY")

        self.embedder = CohereEmbedder(self.cohere_api_key)
        self.storage = QdrantStorage(self.qdrant_url, self.qdrant_api_key)
        self.fetcher = ContentFetcher(self.base_url)
        self.processor = TextProcessor()

    async def process_content(self, incremental: bool = False):
        """Main processing pipeline with optional incremental updates"""
        start_time = asyncio.get_event_loop().time()
        logger.info("Starting content processing pipeline")

        # Fetch all URLs from the site
        urls = self.fetcher.extract_all_urls()
        logger.info(f"Found {len(urls)} URLs to process")

        all_chunks = []
        processed_urls = 0
        failed_urls = 0

        # Process each URL
        for url in urls:
            try:
                logger.info(f"Processing URL: {url}")

                # Fetch content
                html_content = self.fetch_content_with_validation(url)
                if not html_content:
                    logger.warning(f"Failed to fetch content from {url}")
                    failed_urls += 1
                    continue

                # Clean content
                text_content = self.processor.clean_content(html_content)

                # If incremental mode, check if content has changed
                if incremental:
                    content_hash = hashlib.md5(text_content.encode()).hexdigest()
                    if await self._is_content_unchanged(url, content_hash):
                        logger.info(f"Content for {url} unchanged, skipping...")
                        continue
                    logger.info(f"Content for {url} has changed, processing...")

                # Chunk text
                chunks = self.processor.chunk_text(text_content, url)
                logger.info(f"Created {len(chunks)} chunks from {url}")

                all_chunks.extend(chunks)
                processed_urls += 1

            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                failed_urls += 1
                continue

        if not all_chunks:
            logger.info("No new or changed content to process. Pipeline completed.")
            self._log_performance_metrics(start_time, processed_urls, failed_urls, len(all_chunks), 0)
            return

        logger.info(f"Total chunks to process: {len(all_chunks)}")

        # Generate embeddings in batches to respect API limits
        batch_size = 96  # Cohere's max batch size is 96
        all_embeddings = []
        total_tokens = 0

        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_texts = [chunk.text for chunk in batch_chunks]

            logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")

            try:
                embeddings = self.embedder.generate_embeddings(batch_texts)
                all_embeddings.extend(embeddings)

                # Track token usage
                total_tokens += sum(len(text.split()) for text in batch_texts)

            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {str(e)}")
                continue

        if len(all_embeddings) != len(all_chunks):
            logger.error(f"Mismatch: {len(all_embeddings)} embeddings for {len(all_chunks)} chunks")
            return

        # Store embeddings in Qdrant
        logger.info("Storing embeddings in Qdrant")
        self.storage.upsert_embeddings(all_chunks, all_embeddings)

        # Validate retrieval with a sample query
        logger.info("Validating retrieval functionality")
        sample_embedding = all_embeddings[0] if all_embeddings else None
        if sample_embedding:
            results = self.storage.validate_retrieval(sample_embedding)
            logger.info("Retrieval validation results:")
            for i, result in enumerate(results[:3]):  # Show top 3 results
                logger.info(f"Result {i+1}: {result['url']} (score: {result['score']:.3f})")

        # Validate 100% page coverage
        await self._validate_page_coverage(urls)

        # Log performance metrics
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        self._log_performance_metrics(duration, processed_urls, failed_urls, len(all_chunks), total_tokens)

        logger.info("Pipeline completed successfully")

    def fetch_content_with_validation(self, url: str) -> Optional[str]:
        """Fetch content with additional validation for malformed HTML"""
        try:
            return self.fetcher.fetch_content(url)
        except Exception as e:
            logger.warning(f"Error fetching content from {url}: {str(e)}")
            # Try with a more permissive approach for malformed HTML
            try:
                response = requests.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                response.raise_for_status()
                return response.text
            except Exception:
                return None

    async def _validate_page_coverage(self, expected_urls: List[str]):
        """Create validation checks for 100% page coverage"""
        logger.info("Validating page coverage...")

        # Check that all expected URLs have been processed by looking in Qdrant
        processed_urls = set()
        try:
            # Get all unique URLs from the Qdrant collection
            scroll_result = self.storage.client.scroll(
                collection_name=self.storage.collection_name,
                limit=10000  # Assuming we won't have more than 10k URLs initially
            )

            for point in scroll_result[0]:
                url = point.payload.get("url")
                if url:
                    processed_urls.add(url)

            expected_set = set(expected_urls)
            missing_urls = expected_set - processed_urls

            if missing_urls:
                logger.warning(f"Missing content for {len(missing_urls)} URLs: {list(missing_urls)[:5]}...")  # Show first 5
            else:
                logger.info(f"✓ All {len(expected_urls)} URLs have been processed and embedded")

        except Exception as e:
            logger.error(f"Error validating page coverage: {str(e)}")

    def _log_performance_metrics(self, duration: float, processed_urls: int, failed_urls: int, total_chunks: int, total_tokens: int):
        """Log performance metrics and API usage"""
        logger.info("=== PERFORMANCE METRICS ===")
        logger.info(f"Processing duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"Processed URLs: {processed_urls}")
        logger.info(f"Failed URLs: {failed_urls}")
        logger.info(f"Total chunks created: {total_chunks}")
        logger.info(f"Total tokens processed: {total_tokens:,}")
        logger.info(f"Processing rate: {processed_urls/duration*60:.2f} URLs/min if successful")

        if duration > 7200:  # 2 hours in seconds
            logger.warning("Processing took longer than 2 hours!")
        else:
            logger.info("✓ Processing completed within 2-hour timeframe")

        logger.info("===========================")

    async def _is_content_unchanged(self, url: str, content_hash: str) -> bool:
        """Check if content has changed by comparing hashes in Qdrant"""
        # Search for any existing chunks with the same URL and content hash
        try:
            # Search for points with matching URL and content hash
            search_result = self.storage.client.scroll(
                collection_name=self.storage.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="url",
                            match=models.MatchValue(value=url)
                        ),
                        models.FieldCondition(
                            key="content_hash",
                            match=models.MatchValue(value=content_hash)
                        )
                    ]
                ),
                limit=1
            )

            # If we found any matching content, it means the content hasn't changed
            return len(search_result[0]) > 0
        except Exception as e:
            logger.warning(f"Error checking content hash for {url}: {str(e)}")
            # If there's an error, assume content has changed to be safe
            return False

    async def process_incremental_updates(self):
        """Process only changed content using hash comparison"""
        logger.info("Starting incremental update process")
        await self.process_content(incremental=True)


async def main():
    """Main entry point with support for full ingestion and incremental updates"""
    import sys

    pipeline = EmbeddingPipeline()

    # Check command line arguments to determine execution mode
    if len(sys.argv) > 1 and sys.argv[1] == "incremental":
        logger.info("Running in incremental update mode")
        await pipeline.process_incremental_updates()
    else:
        logger.info("Running in full ingestion mode")
        await pipeline.process_content()


if __name__ == "__main__":
    asyncio.run(main())
