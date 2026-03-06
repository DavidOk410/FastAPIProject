# app/settings.py
from pydantic_settings import BaseSettings
from typing import List, Dict

class Settings(BaseSettings):
    # docs
    DOCS: List[Dict[str, str]] = [
        {"doc_id": "fluid-mechanics-module-7", "path": r".\Fluid-Mechanics-Module-7.pdf"},
        {"doc_id": "python-crash-course", "path": r".\Python Crash Course.pdf"},
    ]

    # models
    LLM_MODEL: str = "llama3.1"
    EMBED_MODEL: str = "nomic-embed-text"

    # chunking
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 300

    # pgvector storage  ✅ MUST be annotated
    DATABASE_URL: str = "postgresql://postgres:postgres@127.0.0.1:5433/mydb"
    PG_TABLE: str = "rag_chunks"

    # retrieval
    K_RETRIEVE: int = 6
    METRIC: str = "cosine"  # "cosine" or "l2"

    # indexing
    REBUILD_INDEX: bool = False

settings = Settings()