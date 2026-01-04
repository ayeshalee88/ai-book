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


def get_all_urls(base_url: str) -> List[str]:
    """Extract all URLs from the base site"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    try:
        response = session.get(base_url)
        response.raise_for_status()
        content = response.text
    except Exception as e:
        logger.error(f"Error fetching base URL: {e}")
        return []

    soup = BeautifulSoup(content, 'html.parser')
    links = soup.find_all('a', href=True)

    urls = set()
    for link in links:
        href = link['href']
        full_url = urljoin(base_url, href)
        if base_url in full_url and full_url not in urls:
            urls.add(full_url)

    # Also add the base URL itself
    urls.add(base_url)
    return list(urls)


def extract_text_from_url(url: str) -> str:
    """Extract clean text content from a URL"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    try:
        response = session.get(url)
        response.raise_for_status()
        html_content = response.text
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return ""

    # Clean HTML to extract text
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


def chunk_text(text: str, url: str, max_chunk_size: int = 512) -> List[ContentChunk]:
    """Chunk text into semantic boundaries with metadata"""
    # Split text into sentences first
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    chunk_index = 0

    for sentence in sentences:
        # Check if adding this sentence would exceed the limit
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
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

            current_chunk = sentence + ". "
            chunk_index += 1
        else:
            current_chunk += sentence + ". "

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

    return chunks


def embed(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Cohere API"""
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable is required")

    client = cohere.Client(cohere_api_key)
    model = "embed-multilingual-v3.0"

    response = client.embed(
        texts=texts,
        model=model,
        input_type="search_document"
    )
    return response.embeddings


def create_collection(qdrant_client: QdrantClient, collection_name: str = "ragembeddings"):
    """Create Qdrant collection if it doesn't exist"""
    try:
        qdrant_client.get_collection(collection_name)
        logger.info(f"Collection {collection_name} already exists")
    except:
        # Create collection with appropriate vector size (Cohere embeddings are 1024-dim)
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
        )
        logger.info(f"Created collection {collection_name}")


def save_chunk_to_qdrant(qdrant_client: QdrantClient, chunks: List[ContentChunk],
                        embeddings: List[List[float]], collection_name: str = "ragembeddings"):
    """Save content chunks with embeddings to Qdrant"""
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

    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
    logger.info(f"Upserted {len(points)} embeddings into Qdrant")


async def ingest_book(base_url: str, qdrant_url: str, qdrant_api_key: str):
    """Main ingestion function that orchestrates the entire process"""
    # Initialize Qdrant client
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Create collection
    create_collection(qdrant_client, "ragembeddings")

    # Get all URLs from the site
    urls = get_all_urls(base_url)
    logger.info(f"Found {len(urls)} URLs to process")

    all_chunks = []

    # Process each URL
    for url in urls:
        logger.info(f"Processing URL: {url}")

        # Extract text
        text_content = extract_text_from_url(url)
        if not text_content:
            logger.warning(f"Failed to extract content from {url}")
            continue

        # Chunk text
        chunks = chunk_text(text_content, url)
        logger.info(f"Created {len(chunks)} chunks from {url}")

        all_chunks.extend(chunks)

    if not all_chunks:
        logger.error("No content chunks were created. Exiting.")
        return

    logger.info(f"Total chunks created: {len(all_chunks)}")

    # Generate embeddings in batches to respect API limits
    batch_size = 96  # Cohere's max batch size is 96
    all_embeddings = []

    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_texts = [chunk.text for chunk in batch_chunks]

        logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")

        try:
            embeddings = embed(batch_texts)
            all_embeddings.extend(embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {str(e)}")
            continue

    if len(all_embeddings) != len(all_chunks):
        logger.error(f"Mismatch: {len(all_embeddings)} embeddings for {len(all_chunks)} chunks")
        return

    # Save to Qdrant
    logger.info("Saving embeddings to Qdrant")
    save_chunk_to_qdrant(qdrant_client, all_chunks, all_embeddings, "ragembeddings")

    logger.info("Book ingestion completed successfully")


async def main():
    """Main entry point"""
    # Configuration
    base_url = "https://ai-book-ochre.vercel.app/"
    qdrant_url = os.getenv("https://3e1ff625-70ea-4079-85a5-6f1475e7af8e.europe-west3-0.gcp.cloud.qdrant.io")
    qdrant_api_key = os.getenv("QDRANT-API-KEY")

    if not all([qdrant_url, qdrant_api_key]):
        raise ValueError("Missing required environment variables: QDRANT_URL, QDRANT_API_KEY")

    try:
        await ingest_book(base_url, qdrant_url, qdrant_api_key)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())