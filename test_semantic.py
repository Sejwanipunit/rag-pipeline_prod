from app.ingestion.loader import load_documents
from app.ingestion.chunking.semantic import chunk_semantic_percentile

docs = load_documents()
chunks = chunk_semantic_percentile(docs)
print(f"Total chunks: {len(chunks)}")
print(f"Strategy: {chunks[0].metadata['strategy']}")
print(f"First chunk: {chunks[0].page_content[:200]}")