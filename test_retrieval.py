from app.ingestion.loader import load_documents
from app.ingestion.chunking.recursive import chunk_documents
from app.retrieval.dense import retrieve_dense
from app.retrieval.bm25 import retrieve_bm25

# Load same docs that are in Qdrant
docs = load_documents()
chunks = chunk_documents(docs)

query = "what is this document about?"

print("\n" + "="*50)
print("DENSE RETRIEVAL")
print("="*50)
dense_results = retrieve_dense(query, k=3)

print("\n" + "="*50)
print("BM25 RETRIEVAL")
print("="*50)
bm25_results = retrieve_bm25(query, k=3, documents=chunks)

print("\n" + "="*50)
print("COMPARISON")
print("="*50)
print(f"Dense result 1: {dense_results[0].page_content[:150]}")
print(f"BM25  result 1: {bm25_results[0].page_content[:150]}")