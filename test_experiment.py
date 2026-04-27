from experiments.runner import run_experiment

result = run_experiment(
    doc_path="data/docs/RAG.pdf",
    chunking="recursive",
    retrieval="hybrid",
    queries=[
        "What are the different types of RAG?",
        "How does dense retrieval work?",
        "What is the difference between sparse and dense retrieval?",
    ],
    k=3
)

print("\n📊 RESULTS SUMMARY")
print("="*50)
for r in result["results"]:
    print(f"\nQ: {r['query']}")
    print(f"A: {r['answer'][:200]}")
    print(f"⏱  Latency: {r['latency_seconds']}s")