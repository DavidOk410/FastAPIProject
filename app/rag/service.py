import json
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any

import anyio
from langchain_core.exceptions import OutputParserException

from app.settings import settings

# ✅ правильные импорты: из app.rag.*, а не app.rag.*
from app.rag.loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.vectordb import wipe_persist_dir, build_chroma_from_documents, load_chroma
from app.rag.chain import build_chain

@dataclass
class RAGService:
    chain: Any

    async def ask(self, question: str) -> Dict[str, Any]:
        """Асинхронный вызов цепочки."""
        try:
            out = await self.chain.ainvoke(question)  # RAGResponse (pydantic)
            return out.model_dump()
        except OutputParserException as e:
            raw = getattr(e, "llm_output", "") or str(e)
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            return {"question": question, "answer": raw.strip(), "sources": [], "confidence": 0.0}
        except Exception as e:
            return {"question": question, "answer": f"ERROR: {e}", "sources": [], "confidence": 0.0}


_service: Optional[RAGService] = None


def _build_service_sync() -> RAGService:
    """
    Синхронная сборка сервиса (PDF load + split + Chroma build/load + chain).
    Запускаем это в отдельном thread через anyio.to_thread.run_sync.
    """
    if settings.REBUILD_INDEX:
        wipe_persist_dir(settings.PERSIST_DIR)

    # 1) Пытаемся загрузить существующую Chroma (быстрее)
    try:
        vector_db = load_chroma(
            embed_model=settings.EMBED_MODEL,
            collection_name=settings.COLLECTION_NAME,
            persist_dir=settings.PERSIST_DIR,
        )
        _ = vector_db._collection.count()  # "touch"
    except Exception:
        # 2) Если не удалось — строим заново
        docs = load_pdf(settings.DOC_PATH)
        chunks = split_documents(docs, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        vector_db = build_chroma_from_documents(
            chunks,
            embed_model=settings.EMBED_MODEL,
            collection_name=settings.COLLECTION_NAME,
            persist_dir=settings.PERSIST_DIR,
        )

    chain = build_chain(
        vector_db=vector_db,
        llm_model=settings.LLM_MODEL,
        k=settings.K_RETRIEVE,
        n_queries=settings.N_QUERIES,
    )

    return RAGService(chain=chain)


async def init_service() -> RAGService:
    """
    Инициализация один раз. Сборку делаем в thread, чтобы не блокировать event loop.
    """
    global _service
    if _service is not None:
        return _service

    _service = await anyio.to_thread.run_sync(_build_service_sync)
    return _service


async def get_service() -> RAGService:
    if _service is None:
        return await init_service()
    return _service
