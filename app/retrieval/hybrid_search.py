from typing import List, Tuple, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.ingestion.embedder import load_vector_store
from config import settings

_cross_encoder: Optional[CrossEncoder] = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print(f"   Loading reranker model: {settings.reranker_model}")
        _cross_encoder = CrossEncoder(settings.reranker_model)
    return _cross_encoder


def _dense_search(query: str, k: int) -> List[Tuple[Document, float]]:
    """Retrieve top-k candidates from Qdrant by vector similarity."""
    results = load_vector_store().similarity_search_with_score(query, k=k)
    print(f"   Dense retrieval: fetched {len(results)} candidates from Qdrant")
    return results


def _bm25_search(query: str, corpus: List[Document]) -> List[Tuple[Document, float]]:
    """Score the corpus documents against the query using BM25."""
    if not corpus:
        return []

    tokenized_corpus = [doc.page_content.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())

    ranked = sorted(zip(corpus, scores.tolist()), key=lambda x: x[1], reverse=True)
    print(f"   Sparse retrieval (BM25): scored {len(corpus)} candidates")
    return ranked


def _reciprocal_rank_fusion(
    dense_results: List[Tuple[Document, float]],
    sparse_results: List[Tuple[Document, float]],
    rrf_k: int = 60,
) -> List[Document]:
    """Merge dense and sparse ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, (doc, _) in enumerate(dense_results, start=1):
        key = str(doc.metadata.get("chunk_id", doc.page_content[:100]))
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        doc_map[key] = doc

    for rank, (doc, _) in enumerate(sparse_results, start=1):
        key = str(doc.metadata.get("chunk_id", doc.page_content[:100]))
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    fused = [doc_map[k] for k in sorted_keys]
    print(f"   RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse → {len(fused)} candidates")
    return fused


def _rerank(query: str, docs: List[Document], top_k: int) -> List[Document]:
    """Score (query, document) pairs with a cross-encoder and return the top-k."""
    cross_encoder = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)
    top = [doc for doc, _ in ranked[:top_k]]
    print(f"   Reranking {len(docs)} candidates → returning top {top_k}")
    return top


def hybrid_search(query: str, k: int = 5) -> List[Document]:
    """Full hybrid search pipeline: dense → BM25 → RRF → cross-encoder rerank.

    Args:
        query: The search query string.
        k: Number of final documents to return.

    Returns:
        Top-k Documents ranked by cross-encoder relevance.
    """
    print(f"\n--- Hybrid Search ---")
    print(f"Query: '{query}'")

    dense_results = _dense_search(query, k=settings.dense_retrieval_k)
    dense_docs = [doc for doc, _ in dense_results]
    sparse_results = _bm25_search(query, corpus=dense_docs)
    fused_docs = _reciprocal_rank_fusion(dense_results, sparse_results)
    final_docs = _rerank(query, fused_docs, top_k=k)

    print(f"Hybrid search complete. Returning {len(final_docs)} documents.\n")
    return final_docs
