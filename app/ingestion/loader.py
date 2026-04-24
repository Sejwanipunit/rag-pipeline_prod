import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)

def load_documents(data_dir: str = "data/docs") -> List[Document]:
    """
    Load all documents from the data directory.
    Supports PDF and text files.
    """
    docs = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
    
    #Load PDFs
    pdf_files = list(data_path.glob("**/*.pdf"))
    for pdf_path in pdf_files:
        print(f"Loading PDF: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
        
    
    #Load text files
    txt_files = list(data_path.glob("**/*.txt"))
    for txt_path in txt_files:
        print(f"Loading text file: {txt_path.name}")
        loader = TextLoader(str(txt_path), encoding="utf-8")
        docs.extend(loader.load())
        
    print(f"\n✅ Loaded {len(docs)} document pages from {len(pdf_files)} PDFs and {len(txt_files)} text files")
    return docs


def load_single_file(file_path: str) -> List[Document]:
    """
    Load a single file by path."""
    
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(file_path)
    elif path.suffix == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} document pages from {path.name}")
    return docs

    
      