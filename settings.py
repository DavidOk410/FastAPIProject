from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DOC_PATH: str = "Fluid-Mechanics-Module-7.pdf"

    LLM_MODEL: str = "llama3.1"
    EMBED_MODEL: str = "nomic-embed-text"

    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 300

    COLLECTION_NAME: str = "simple-rag"
    PERSIST_DIR: str = "./chroma_db"  # сохранение индекса на диск

    K_RETRIEVE: int = 6
    N_QUERIES: int = 5

    # если True — при старте всегда пересоздаём индекс с нуля
    REBUILD_INDEX: bool = False

settings = Settings()
