import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
from flask import Flask, render_template, request, jsonify
from experiments.runner import run_experiment, compare_experiments

app = Flask(__name__)

# ── Available strategies ──────────────────────────────────────
CHUNKING_OPTIONS = [
    ("recursive",            "Recursive (default)"),
    ("fixed_256",            "Fixed — 256 chars"),
    ("fixed_512",            "Fixed — 512 chars"),
    ("fixed_1024",           "Fixed — 1024 chars"),
    ("semantic_percentile",  "Semantic — Percentile"),
    ("semantic_std",         "Semantic — Std Deviation"),
]

RETRIEVAL_OPTIONS = [
    ("dense",   "Dense (vector search)"),
    ("bm25",    "BM25 (keyword search)"),
    ("hybrid",  "Hybrid (BM25 + Dense + Reranker)"),
]


@app.route("/")
def index():
    return render_template(
        "index.html",
        chunking_options=CHUNKING_OPTIONS,
        retrieval_options=RETRIEVAL_OPTIONS,
    )


@app.route("/run", methods=["POST"])
def run():
    data = request.form

    chunking  = data.get("chunking", "recursive")
    retrieval = data.get("retrieval", "hybrid")
    queries   = [q.strip() for q in data.get("queries", "").split("\n") if q.strip()]
    k         = int(data.get("k", 3))

    if not queries:
        return jsonify({"error": "Please enter at least one query"}), 400

    try:
        result = run_experiment(
            doc_path="data/docs/RAG.pdf",
            chunking=chunking,
            retrieval=retrieval,
            queries=queries,
            k=k,
        )
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/results")
def results():
    all_results = compare_experiments()
    return render_template("results.html", experiments=all_results)


@app.route("/api/results")
def api_results():
    return jsonify(compare_experiments())


if __name__ == "__main__":
    app.run(debug=True, port=5001)