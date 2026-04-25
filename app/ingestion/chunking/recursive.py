from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller overlapping chunks.
    Uses RecursiveCharacterTextSplitter — tries to split on
    paragraphs first, then sentences, then words. This keeps
    semantic meaning intact as much as possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,        # 512 chars per chunk
        chunk_overlap=settings.chunk_overlap,  # 50 chars overlap
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    chunks = splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i  # Add chunk ID to metadata
        chunk.metadata["chunk_size"] = len(chunk.page_content)  # Add chunk size to metadata

    print(f"✅ Split {len(documents)} pages into {len(chunks)} chunks (chunk size: {settings.chunk_size}, overlap: {settings.chunk_overlap})")
    print(f"   - Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks):.2f} characters")
    return chunks

def inspect_chunks(chunks: List[Document], n: int = 3):
    """Helper to preview first n chunks — useful during development."""
    print(f"\n🔍 Previewing first {n} chunks:\n")
    for i, chunk in enumerate(chunks[:n]):
        print(f"--- Chunk {i} ---")
        print(f"Size: {chunk.metadata['chunk_size']} chars")
        print(f"Source: {chunk.metadata.get('source', 'unknown')}")
        print(f"Content: {chunk.page_content[:150]}...")
        print()