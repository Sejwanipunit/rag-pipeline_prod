# app/retrieval/bm25.py
from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


# Global cache — BM25 index built once, reused across calls
_bm25_retriever = None


def build_bm25_index(documents: List[Document]) -> BM25Retriever:
    """
    Build BM25 index from documents.
    Call this once after ingestion — index lives in memory.
    """
    global _bm25_retriever

    retriever = BM25Retriever.from_documents(
        documents,
        k=5
    )
    _bm25_retriever = retriever
    print(f"✅ BM25 index built from {len(documents)} chunks")
    return retriever


def retrieve_bm25(query: str, k: int = 5, documents: List[Document] = None) -> List[Document]:
    """
    Pure BM25 keyword retrieval.
    Great for exact matches — model numbers, names, specific terms.
    Requires documents to build index if not already built.
    """
    global _bm25_retriever

    if _bm25_retriever is None:
        if documents is None:
            raise ValueError(
                "BM25 index not built yet. Pass documents on first call."
            )
        build_bm25_index(documents)

    _bm25_retriever.k = k
    results = _bm25_retriever.invoke(query)

    print(f"🔍 BM25 retrieval: found {len(results)} chunks for: '{query}'")
    for i, doc in enumerate(results):
        print(f"   [{i+1}] chunk_id={doc.metadata.get('chunk_id')} | {doc.page_content[:80]}...")

    return results