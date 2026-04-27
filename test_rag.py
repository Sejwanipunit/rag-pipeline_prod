from app.ingestion.loader import load_documents
from app.ingestion.chunking.recursive import chunk_documents
from app.retrieval.hybrid_search import hybrid_search
from app.generation.chain import generate_answer

query = "What are the different types of RAG?"

# Retrieve
docs = hybrid_search(query, k=3)

# Generate
result = generate_answer(query, docs)

print("\n" + "="*50)
print("QUERY:", result["query"])
print("="*50)
print("ANSWER:", result["answer"])
print("\nSOURCES:")
for s in result["sources"]:
    print(f"  - {s['source']} | page {s['page']} | chunk {s['chunk_id']}")