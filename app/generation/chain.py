from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from config import settings


# ── Prompt Template ──────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY 
the context provided below.

If the answer is not present in the context, respond with:
"I don't know based on the provided context."

Do not make up information. Do not use prior knowledge.

Context:
{context}

Question: {question}

Answer:
""")


def get_llm():
    """
    Load the LLM — Groq via OpenAI-compatible endpoint.
    Temperature 0.0 = deterministic answers, no creativity.
    For RAG we want factual, consistent responses.
    """
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.max_tokens,
        openai_api_key=settings.groq_api_key,
        openai_api_base=settings.groq_base_url,
    )


def format_context(docs: List[Document]) -> str:
    """
    Convert list of Document objects into a single context string.
    Each chunk is numbered and separated clearly so the LLM
    can distinguish between different source chunks.
    """
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Chunk {i} | Source: {source} | Page: {page}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(context_parts)


def generate_answer(query: str, docs: List[Document]) -> dict:
    """
    Generate an answer from retrieved docs using the LLM.
    
    Returns a dict with:
    - answer: the LLM response
    - query: original question
    - context: formatted chunks used
    - num_chunks: how many chunks were used
    """
    llm = get_llm()
    context = format_context(docs)

    # Build the chain: prompt → LLM → parse output as string
    chain = RAG_PROMPT | llm | StrOutputParser()

    print(f"\n🤖 Generating answer for: '{query}'")
    print(f"   Using {len(docs)} chunks as context")

    answer = chain.invoke({
        "context": context,
        "question": query
    })

    return {
        "query": query,
        "answer": answer,
        "context": context,
        "num_chunks": len(docs),
        "sources": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "chunk_id": doc.metadata.get("chunk_id", "?"),
            }
            for doc in docs
        ]
    }