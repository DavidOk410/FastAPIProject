from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DOC_PATH: str = r".\Fluid-Mechanics-Module-7.pdf"

    LLM_MODEL: str = "llama3.1"
    EMBED_MODEL: str = "nomic-embed-text"

    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 300

    COLLECTION_NAME: str = "simple-rag"
    PERSIST_DIR: str = "./chroma_db_v5"   # ✅ ВОТ ТАК
    K_RETRIEVE: int = 6
    N_QUERIES: int = 5

    REBUILD_INDEX: bool = False # ✅ если хочешь пересобрать

settings = Settings()
