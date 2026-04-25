from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter


def chunk_fixed(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 50) -> List[Document]:
    """
    Fixed-size chunking - splits at exact  character count.
    No consideration for sentence or boundaries.
    Fast and simple but lower retrieval quality vs recursive chunking.
    """
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="",
        length_function=len,
    )
    
    chunks = splitter.split_documents(documents)
    
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i  # Add chunk ID to metadata
        chunk.metadata["chunk_size"] = len(chunk.page_content)  # Add chunk size to metadata
        chunk.metadata["strategy"] = f"fixed_{chunk_size}_{chunk_overlap}"  # Add strategy info to metadata
        
    print(f"✅ Fixed chunking: {len(documents)} pages -> {len(chunks)} chunks (size:{chunk_size}")
    return chunks

def chunk_fixed_256(documents: List[Document]) -> List[Document]:
    """Preset - small chunks, high precision"""
    return chunk_fixed(documents, chunk_size=256, chunk_overlap=25)

def chunk_fixed_512(documents: List[Document]) -> List[Document]:
    """Preset — medium chunks, balanced."""
    return chunk_fixed(documents, chunk_size=512, chunk_overlap=50)


def chunk_fixed_1024(documents: List[Document]) -> List[Document]:
    """Preset — large chunks, more context."""
    return chunk_fixed(documents, chunk_size=1024, chunk_overlap=100)



    