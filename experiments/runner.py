import json
import time
import os
from typing import List
from datetime import datetime
from langchain_core.documents import Document

from app.ingestion.loader import load_documents
from app.ingestion.embedder import embed_and_store, load_vector_store
from app.ingestion.chunking.recursive import chunk_documents as chunk_recursive
from app.ingestion.chunking.fixed import chunk_fixed_256, chunk_fixed_512, chunk_fixed_1024
from app.ingestion.chunking.semantic import chunk_semantic_percentile, chunk_semantic_std
from app.retrieval.dense import retrieve_dense
from app.retrieval.bm25 import retrieve_bm25
from app.retrieval.hybrid_search import hybrid_search
from app.generation.chain import generate_answer

# Mapping string names to actual functions.

CHUNKING_STRATEGIES = {
    "recursive":        chunk_recursive,
    "fixed_256":        chunk_fixed_256,
    "fixed_512":        chunk_fixed_512,
    "fixed_1024":       chunk_fixed_1024,
    "semantic_percentile": chunk_semantic_percentile,
    "semantic_std":     chunk_semantic_std,
}

RETRIEVAL_STRATEGIES = {
    "dense":   lambda query, k, chunks: retrieve_dense(query, k),
    "bm25":    lambda query, k, chunks: retrieve_bm25(query, k, documents=chunks),
    "hybrid":  lambda query, k, chunks: hybrid_search(query, k),
}


# Core Run

def run_experiment(
    doc_path: str,
    chunking: str,
    retrieval: str,
    queries: List[str],
    k: int = 5,
    collection_name: str = None,
) -> dict:
    """
    Run a full RAG experiment with given strategies.

    Args:
        doc_path:   path to PDF or txt file
        chunking:   strategy name — "recursive", "fixed_512", "semantic_percentile" etc
        retrieval:  strategy name — "dense", "bm25", "hybrid"
        queries:    list of test questions
        k:          number of chunks to retrieve per query
        collection_name: optional custom Qdrant collection name

    Returns:
        experiment result dict saved to experiments/results/
    """

    print(f"\n{'='*60}")
    print(f"🧪 EXPERIMENT")
    print(f"   Chunking:  {chunking}")
    print(f"   Retrieval: {retrieval}")
    print(f"   Queries:   {len(queries)}")
    print(f"{'='*60}\n")

    # Validate
    if chunking not in CHUNKING_STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: '{chunking}'. "
                        f"Choose from: {list(CHUNKING_STRATEGIES.keys())}")

    if retrieval not in RETRIEVAL_STRATEGIES:
        raise ValueError(f"Unknown retrieval strategy: '{retrieval}'. "
                        f"Choose from: {list(RETRIEVAL_STRATEGIES.keys())}")

    experiment_start = time.time()

    # Load Documents
    print("📄 Step 1: Loading documents...")
    docs = load_documents(os.path.dirname(doc_path))

    # Chunking with method selected
    print(f"\n✂️  Step 2: Chunking with '{chunking}'...")
    chunk_fn = CHUNKING_STRATEGIES[chunking]
    chunks = chunk_fn(docs)

    # Embed and store
    print(f"\n📦 Step 3: Embedding {len(chunks)} chunks...")
    embed_and_store(chunks)

    # Run query
    print(f"\n🔍 Step 4: Running {len(queries)} queries with '{retrieval}' retrieval...")
    retrieval_fn = RETRIEVAL_STRATEGIES[retrieval]

    query_results = []
    for i, query in enumerate(queries, 1):
        print(f"\n   Query {i}/{len(queries)}: {query}")

        query_start = time.time()

        # Retrieve
        retrieved_chunks = retrieval_fn(query, k, chunks)

        # Generate answer
        result = generate_answer(query, retrieved_chunks)

        query_latency = time.time() - query_start

        query_results.append({
            "query": query,
            "answer": result["answer"],
            "latency_seconds": round(query_latency, 2),
            "num_chunks_retrieved": len(retrieved_chunks),
            "sources": result["sources"],
            "retrieved_context": [
                chunk.page_content[:200]
                for chunk in retrieved_chunks
            ]
        })

    total_time = time.time() - experiment_start
    avg_latency = sum(r["latency_seconds"] for r in query_results) / len(query_results)

    # Result dict to save
    experiment_result = {
        "experiment_id": f"{chunking}__{retrieval}__{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "doc_path": doc_path,
            "chunking_strategy": chunking,
            "retrieval_strategy": retrieval,
            "k": k,
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(
                len(c.page_content) for c in chunks
            ) // len(chunks),
        },
        "performance": {
            "total_time_seconds": round(total_time, 2),
            "avg_query_latency_seconds": round(avg_latency, 2),
        },
        "results": query_results,
    }

    # Save results to JSON
    save_results(experiment_result)

    print(f"\n{'='*60}")
    print(f"✅ Experiment complete!")
    print(f"   Total time:   {total_time:.1f}s")
    print(f"   Avg latency:  {avg_latency:.2f}s per query")
    print(f"   Results saved to experiments/results/")
    print(f"{'='*60}")

    return experiment_result


def save_results(result: dict):
    """Save experiment results to JSON file."""
    os.makedirs("experiments/results", exist_ok=True)
    filename = f"experiments/results/{result['experiment_id']}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)
    print(f"💾 Results saved: {filename}")


def compare_experiments(results_dir: str = "experiments/results") -> List[dict]:
    """
    Load all saved experiment results for comparison.
    Used by Flask UI to show comparison table.
    """
    results = []
    if not os.path.exists(results_dir):
        return results

    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            with open(os.path.join(results_dir, filename)) as f:
                results.append(json.load(f))

    # Sort by timestamp newest first
    results.sort(key=lambda x: x["timestamp"], reverse=True)
    return results