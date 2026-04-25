from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import settings


def get_embeddings():
    """
    Load the sentence-transformer embedding model.
    all-MiniLM-L6-v2 is small (80MB), fast, and good quality.
    Downloads automatically on first run, cached after that.
    """
    print("🔍 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ Embedding model loaded")
    return embeddings


def create_collection(client: QdrantClient, collection_name: str):
    """Create Qdrant collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Created Qdrant collection: {collection_name}")
    else:
        print(f"✅ Collection already exists: {collection_name}")


def embed_and_store(chunks: List[Document]) -> QdrantVectorStore:
    """
    Embed all chunks and store them in Qdrant.
    Returns the vector store for later retrieval.
    """
    embeddings = get_embeddings()

    client = QdrantClient(url=settings.qdrant_url)
    create_collection(client, settings.qdrant_collection)

    print(f"📦 Embedding {len(chunks)} chunks into collection '{settings.qdrant_collection}'...")

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )

    print(f"✅ Upserted {len(chunks)} chunks successfully.")
    return vector_store


def load_vector_store() -> QdrantVectorStore:
    """
    Connect to existing Qdrant collection.
    Used when data is already ingested — no re-embedding needed.
    """
    embeddings = get_embeddings()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )
    print("✅ Connected to existing Qdrant collection")
    return vector_store