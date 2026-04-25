from app.ingestion.loader import load_documents
from app.ingestion.chunking.recursive import chunk_documents, inspect_chunks
from app.ingestion.embedder import embed_and_store

if __name__ == "__main__":
    docs = load_documents("data/docs")
    chunks = chunk_documents(docs)
    inspect_chunks(chunks, n=3)
    embed_and_store(chunks)
    print("✅ Ingestion complete. Run hybrid_search to test retrieval.")
