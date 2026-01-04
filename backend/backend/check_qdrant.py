from qdrant_client import QdrantClient
from src.config.settings import settings

client = QdrantClient(
    url="https://3e1ff625-70ea-4079-85a5-6f1475e7af8e.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.pQlo8rzjHRVm0Lb1kYX0CoCloKG-nJ5tal6BGG8DMUQ"
)

collections = client.get_collections()
print(collections)

info = client.get_collection("book_content")
print("Collection info:", info)