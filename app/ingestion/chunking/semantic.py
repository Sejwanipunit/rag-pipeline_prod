from typing import List
from langchain_core.documents import Document
from config import settings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


def get_semantic_splitter(breakpoint_type: str = "percentile", threshold: float = 95.0):
    """Create a semantic chunker with specified breakpoint strategy.
    breakpoint_type options:
    - "percentile" : cut where similarity drop is in top X
    - "standard_deviation" : cut where similarity drops exceeds X standard deviations
    - "interquartile" : cut using IQR statistical method
    
    Lower threshold = more cuts = smaller chunks
    Higher threshold = fewer cuts = larger chunks
    """
    
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"}, # Use CPU for embedding generation
        encode_kwargs={"normalize_embeddings": True}
    )

    splitter = SemanticChunker(
        embeddings = embeddings,
        breakpoint_threshold_type = breakpoint_type,
        breakpoint_threshold_amount = threshold,
    )
    
    return splitter

def chunk_semantic(
    documents: List[Document],
    breakpoint_type: str = "percentile",
    threshold: float = 95.0
) -> List[Document]:
    """
    Semantic chunking — splits at topic boundaries detected via embeddings.
    Slower than fixed/recursive but much higher retrieval quality.
    """
    print(f"🔍 Semantic chunking (breakpoint={breakpoint_type}, threshold={threshold})...")
    print("   Note: this is slower — embeds every sentence during splitting")

    splitter = get_semantic_splitter(breakpoint_type, threshold)
    chunks = splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i  # Add chunk ID to metadata
        chunk.metadata["chunk_size"] = len(chunk.page_content)  # Add chunk size to metadata
        chunk.metadata["strategy"] = f"semantic_{breakpoint_type}"  # Add strategy info to metadata
        
    print(f"✅ Semantic chunking: {len(documents)} pages → {len(chunks)} chunks")
    avg_size = sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 0
    print(f"   Avg chunk size: {avg_size} chars")

    return chunks

def chunk_semantic_percentile(documents: List[Document]) -> List[Document]:
    """Preset — percentile method, good general purpose."""
    return chunk_semantic(documents, "percentile", 95.0)


def chunk_semantic_std(documents: List[Document]) -> List[Document]:
    """Preset — standard deviation method, more aggressive cuts."""
    return chunk_semantic(documents, "standard_deviation", 1.5)