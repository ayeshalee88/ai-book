from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
import uuid
from backend.src.config.settings import settings



client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")  # Better, reliable, same size (384 dims)

collection_name = "book_content"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

chunks = [
    "Hello from Physical AI & Humanoid Robotics",
    "Physical AI & Humanoid Robotics",
    "Bridging the gap between Digital AI & Physical Robotics",
    "Start the Course - 13 Weeks",
    "Master Physical AI principles, embodied intelligence, humanoid robotics, and conversational AI through our structured 13-week curriculum.",
    "Explore the design principles, kinematics, and control systems behind humanoid robots that bridge digital and physical intelligence.",
    "Get hands-on experience with tutorials, code examples, and real-world case studies from industry leaders.",
    "Learning Outcomes: Master Physical AI principles, embodied intelligence, humanoid robotics development.",
    "Humanoid Robotics: Design principles, kinematics, control systems.",
    "Practical Implementation: Hands-on tutorials, code examples, real-world case studies.",
]

embeddings = list(embedding_model.embed(chunks))

points = []
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    points.append(PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding.tolist(),
        #payload={"content": chunk, "metadata": {"chunk_id": i, "source": "ai-book-ochre.vercel.app"}}
        payload = {
            "content": chunk['content'],

             # ✅ chapter-aware fields
            "chapter_name": chunk['metadata']['chapter_name'],
            "chapter_number": chunk['metadata']['chapter_number'],

            # page info
            "page_title": chunk['metadata']['page_title'],
            "page_url": chunk['metadata']['page_url'],
            "source": chunk['metadata']['source'],

            # chunk info
            "chunk_id": chunk['metadata']['chunk_id'],
            "total_chunks": chunk['metadata']['total_chunks']
        }
    ))

client.upload_points(collection_name=collection_name, points=points)
print("SUCCESS! All your course content loaded with a reliable model. The AI is ready!")
