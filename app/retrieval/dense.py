# app/retrieval/dense.py
from typing import List
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from app.ingestion.embedder import load_vector_store


def retrieve_dense(query: str, k: int = 5) -> List[Document]:
    """
    Pure dense retrieval — semantic similarity search via Qdrant.
    Good for meaning-based queries but misses exact keyword matches.
    """
    vector_store = load_vector_store()

    results = vector_store.similarity_search(
        query=query,
        k=k
    )

    print(f"🔍 Dense retrieval: found {len(results)} chunks for query: '{query}'")
    for i, doc in enumerate(results):
        print(f"   [{i+1}] score chunk_id={doc.metadata.get('chunk_id')} | {doc.page_content[:80]}...")

    return results


def retrieve_dense_with_scores(query: str, k: int = 5):
    """Returns chunks with their similarity scores — useful for evaluation."""
    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(
        query=query,
        k=k
    )

    print(f"🔍 Dense retrieval with scores:")
    for doc, score in results:
        print(f"   score={score:.4f} | {doc.page_content[:80]}...")

    return results