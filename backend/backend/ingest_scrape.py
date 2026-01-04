
"""
Web scraper to extract all content from ai-book-ochre.vercel.app and ingest into Qdrant
"""
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
import uuid
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

# Update import path based on your structure
try:
    from backend.src.config.settings import settings
except:
    from src.config.settings import settings

# Initialize clients
client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
    timeout=600  # Increased to 10 minutes
)
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
collection_name = "book_content"


def extract_chapter_from_url(url: str, title: str):
    """
    Try to infer chapter number/name from URL or title
    """
    # Example patterns:
    # /docs/chapter-5-introduction
    # /docs/chapter5
    # Chapter 5: Simulation Environments

    match = re.search(r'chapter[-\s]?(\d+)', url.lower())
    if not match:
        match = re.search(r'chapter[-\s]?(\d+)', title.lower())

    if match:
        num = match.group(1)
        return {
            "chapter_number": int(num),
            "chapter_name": f"Chapter {num}"
        }

    return {
        "chapter_number": None,
        "chapter_name": None
    }

def get_all_pages(base_url: str) -> list:
    """Discover all pages/sections on the website"""
    visited = set()
    to_visit = [base_url]
    pages = []
    
    while to_visit:
        url = to_visit.pop(0)
        
        if url in visited:
            continue
        
        # Only process URLs from the same domain
        if not url.startswith(base_url):
            continue
            
        visited.add(url)
        
        try:
            print(f"Discovering: {url}")
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            pages.append({
                'url': url,
                'soup': soup
            })
            
            # Find all internal links
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                
                # Only follow links within the same domain
                if full_url.startswith(base_url) and full_url not in visited:
                    to_visit.append(full_url)
            
            time.sleep(0.5)  # Be nice to the server
            
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
    
    return pages


def extract_content(soup, url: str) -> dict:
    """Extract meaningful content from a page"""
    
    # Remove script, style, and navigation elements
    for element in soup(['script', 'style', 'nav', 'header', 'footer']):
        element.decompose()
    
    # Try to find the main content area
    main_content = (
        soup.find('main') or 
        soup.find('article') or 
        soup.find('div', class_=['content', 'main-content', 'post-content']) or
        soup.find('body')
    )
    
    if not main_content:
        return None
    
    # Extract title
    title = soup.find('h1')
    title_text = title.get_text(strip=True) if title else url.split('/')[-1]
    
    # Extract all text content
    text = main_content.get_text(separator='\n', strip=True)
    
    # Extract headings for structure
    headings = []
    for heading in main_content.find_all(['h1', 'h2', 'h3', 'h4']):
        headings.append({
            'level': heading.name,
            'text': heading.get_text(strip=True)
        })
    
    return {
        'url': url,
        'title': title_text,
        'content': text,
        'headings': headings
    }


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def scrape_and_ingest(base_url: str):
    """Main function to scrape website and ingest into Qdrant"""
    
    print(f"🕷️  Starting to scrape: {base_url}")
    print("=" * 60)
    
    # Step 1: Discover all pages
    print("\n📍 Step 1: Discovering all pages...")
    pages = get_all_pages(base_url)
    print(f"✓ Found {len(pages)} pages")
    
    # Step 2: Extract content from each page
    print("\n📄 Step 2: Extracting content...")
    all_content = []
    for page_data in pages:
        content = extract_content(page_data['soup'], page_data['url'])
        if content and len(content['content']) > 100:  # Skip very short pages
            all_content.append(content)
            print(f"✓ Extracted: {content['title'][:50]}...")
    
    print(f"✓ Extracted content from {len(all_content)} pages")
    
    # Step 3: Delete and recreate collection
    print("\n🗑️  Step 3: Resetting Qdrant collection...")
    try:
        client.delete_collection(collection_name)
        print(f"✓ Deleted existing collection")
    except:
        pass
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"✓ Created fresh collection: {collection_name}")
    
    # Step 4: Chunk and prepare for ingestion
    print("\n✂️  Step 4: Chunking content...")
    all_chunks = []
    
    for page in all_content:
        chapter_info = extract_chapter_from_url(page['url'], page['title'])
        chunks = chunk_text(page['content'], chunk_size=500, overlap=50)

        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append({
            'content': chunk,
            'metadata': {
                'source': base_url,
                'page_url': page['url'],
                'page_title': page['title'],

                # ✅ NEW (CRITICAL)
                'chapter_name': chapter_info["chapter_name"],
                'chapter_number': chapter_info["chapter_number"],

                'chunk_id': chunk_idx,
                'total_chunks': len(chunks)
            }
        })

    
    print(f"✓ Created {len(all_chunks)} chunks")
    
    # Step 5: Generate embeddings and upload
    print("\n🚀 Step 5: Generating embeddings and uploading...")
    batch_size = 25  # Reduced from 50 to 25 for better reliability
    max_retries = 5
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        
        # Generate embeddings
        texts = [chunk['content'] for chunk in batch]
        embeddings = list(embedding_model.embed(texts))
        
        # Create points
        points = []
        for chunk, embedding in zip(batch, embeddings):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "content": chunk['content'],
                    "metadata": chunk['metadata']
                }
            ))
        
        # Upload to Qdrant with retry logic
        retry_count = 0
        while retry_count < max_retries:
            try:
                client.upload_points(collection_name=collection_name, points=points)
                break  # Success, exit retry loop
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"\n❌ Failed to upload batch after {max_retries} attempts")
                    print(f"Error: {str(e)}")
                    print(f"Skipping batch {i} to {i + len(batch)}")
                    break
                else:
                    print(f"⚠️  Upload failed (attempt {retry_count}/{max_retries}). Retrying in {retry_count * 2} seconds...")
                    time.sleep(retry_count * 2)  # Exponential backoff
        
        progress = ((i + len(batch)) / len(all_chunks)) * 100
        print(f"✓ Progress: {progress:.1f}% ({i + len(batch)}/{len(all_chunks)} chunks)")
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS! Book content ingested into Qdrant!")
    print("=" * 60)
    print(f"📊 Statistics:")
    print(f"   - Pages scraped: {len(all_content)}")
    print(f"   - Total chunks: {len(all_chunks)}")
    print(f"   - Collection: {collection_name}")
    print(f"   - Ready to answer questions about: Physical AI & Humanoid Robotics")
    print("\n🎯 Try asking:")
    print("   - 'What is in chapter 2?'")
    print("   - 'Explain physical AI'")
    print("   - 'What are the learning outcomes?'")
    print("=" * 60)


if __name__ == "__main__":
    base_url = "https://ai-book-ochre.vercel.app/"
    
    try:
        scrape_and_ingest(base_url)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()