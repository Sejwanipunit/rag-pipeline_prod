from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import settings


def embed_and_store(chunks: List[Document]) -> QdrantVectorStore:
    """Embed chunks with HuggingFace sentence-transformers and store in Qdrant.

    Recreates the collection on every call (delete + create), so running
    ingest.py multiple times always gives a clean slate.
    """
    print(f"\n📦 Embedding {len(chunks)} chunks into collection '{settings.qdrant_collection}'...")

    embeddings = HuggingFaceEmbeddings(model_name=settings.embeddings_model)
    vector_size = len(embeddings.embed_query("test"))

    client = QdrantClient(url=settings.qdrant_url)

    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"   Collection '{settings.qdrant_collection}' created. Vector size: {vector_size}, Distance: COSINE")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )

    vector_store.add_documents(chunks)
    print(f"✅ Upserted {len(chunks)} chunks successfully.\n")

    return vector_store


def get_vector_store() -> QdrantVectorStore:
    """Return a QdrantVectorStore connected to the existing collection (read-only)."""
    embeddings = HuggingFaceEmbeddings(model_name=settings.embeddings_model)
    client = QdrantClient(url=settings.qdrant_url)
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )
