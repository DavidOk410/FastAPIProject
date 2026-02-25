# app/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DOCS: list[dict] = [
            {"doc_id": "fluid-mechanics-module-7", "path": r".\Fluid-Mechanics-Module-7.pdf"},
            {"doc_id": "python-crash-course", "path": r".\Python Crash Course.pdf"},
    ]

    LLM_MODEL: str = "llama3.1"
    EMBED_MODEL: str = "nomic-embed-text"

    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 300

    # pgvector storage
    DATABASE_URL: str = "dbname=mydb user=postgres password=postgres host=localhost port=5433"
    PG_TABLE: str = "rag_chunks"

    # retrieval
    K_RETRIEVE: int = 6
    METRIC: str = "cosine"  # "cosine" or "l2"

    # индексация
    REBUILD_INDEX: bool = False

settings = Settings()
