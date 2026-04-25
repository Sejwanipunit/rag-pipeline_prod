from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    groq_api_key: str
    groq_base_url: str = "https://api.groq.com/openai/v1"
    
    #LLM Settings
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.0
    max_tokens: int = 1024
    
    #Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    #Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_docs"
    
    #LangSmith
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "true"
    langchain_project: str = "rag-pipeline"
    
    #Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    dense_retrieval_k: int = 20
    reranker_top_k: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"
        
settings = Settings()