import json
import re
import logging
from pathlib import Path
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


log = logging.getLogger("rag.service")

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
            friendly = _friendly_runtime_error(e)
            log.exception("RAG ask failed: %s", e)
            return {
                "question": question,
                "answer": friendly,
                "sources": [],
                "confidence": 0.0,
            }


def _friendly_runtime_error(error: Exception) -> str:
    """Return actionable error messages for common runtime backend issues."""
    raw = str(error).strip()
    normalized = raw.lower()

    ollama_hints = (
        "all connection attempts failed",
        "connection refused",
        "connecterror",
        "failed to connect",
        "max retries exceeded",
    )
    if any(hint in normalized for hint in ollama_hints):
        return (
            "LLM backend is unavailable. Start Ollama and ensure the model is installed "
            f"(model: '{settings.LLM_MODEL}'). Original error: {raw}"
        )

    return f"Unexpected RAG runtime error: {raw}"


_service: Optional[RAGService] = None


def _resolve_doc_path(raw_path: str) -> Path:
    """Resolve DOC_PATH robustly regardless of current working directory."""
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path

    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path.cwd() / path,
        project_root / path,
        project_root / "app" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"PDF file not found for DOC_PATH='{raw_path}'. CWD='{Path.cwd()}'. Tried: {tried}"
    )


def _build_service_sync() -> RAGService:
    """
    Синхронная сборка сервиса:
    - если есть индекс -> load
    - если нет -> build
    - если REBUILD_INDEX=True -> build (но НЕ пытаемся wipe, если папка занята; используем новую папку или ручной stop)
    """
    persist_path = Path(settings.PERSIST_DIR)
    persist_exists = persist_path.exists()
    log.info("REBUILD_INDEX=%s, PERSIST_DIR=%s, persist_exists=%s",
             settings.REBUILD_INDEX, settings.PERSIST_DIR, persist_exists)

    resolved_doc_path = _resolve_doc_path(settings.DOC_PATH)

    # --- Decide: load vs build ---
    should_build = settings.REBUILD_INDEX or (not persist_exists)

    if should_build:
        # Если rebuild, и папка существует — пробуем wipe.
        # Если wipe не выходит из-за WinError32 — НЕ падаем: просто говорим, что нельзя rebuild пока база используется.
        if persist_exists:
            try:
                wipe_persist_dir(settings.PERSIST_DIR)
                persist_exists = False
            except Exception as e:
                # Это как раз твой WinError 32
                raise RuntimeError(
                    f"Cannot rebuild index because persist dir is in use: {settings.PERSIST_DIR}. "
                    "Stop other uvicorn/python processes or change PERSIST_DIR to a new folder."
                ) from e

        log.info("Building index from PDF: %s", resolved_doc_path)
        docs = load_pdf(str(resolved_doc_path))
        if not docs:
            raise RuntimeError("PDF loader returned 0 pages (check DOC_PATH).")

        chunks = split_documents(docs, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        log.info("Chunks created: %d", len(chunks))
        if not chunks:
            raise RuntimeError("Splitter returned 0 chunks (PDF may have no extractable text).")

        vector_db = build_chroma_from_documents(
            chunks,
            embed_model=settings.EMBED_MODEL,
            collection_name=settings.COLLECTION_NAME,
            persist_dir=settings.PERSIST_DIR,
        )
        log.info("Index ready. count=%s", vector_db._collection.count())

    else:
        log.info("Loading existing index from %s", settings.PERSIST_DIR)
        vector_db = load_chroma(
            embed_model=settings.EMBED_MODEL,
            collection_name=settings.COLLECTION_NAME,
            persist_dir=settings.PERSIST_DIR,
        )
        count = vector_db._collection.count()
        log.info("Loaded index. count=%s", count)
        if count == 0:
            raise RuntimeError("Persisted index is empty. Set REBUILD_INDEX=True to rebuild.")

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
