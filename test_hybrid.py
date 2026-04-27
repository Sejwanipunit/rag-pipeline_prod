from app.retrieval.hybrid_search import hybrid_search

queries = [
    "What are the different types of RAG?",
    "How does dense retrieval work?",
    "What is the difference between sparse and dense retrieval?",
]

for query in queries:
    print(f"\n{'='*50}")
    print(f"Query: {query}")
    print('='*50)
    results = hybrid_search(query, k=3)
    for i, doc in enumerate(results):
        print(f"\n[{i+1}] {doc.page_content[:200]}")