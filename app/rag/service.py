import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

import anyio
from langchain_core.exceptions import OutputParserException

from app.settings import settings
from app.rag.loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.pgvector import PgVectorStore
from app.rag.chain import build_chain

log = logging.getLogger("rag.service")


@dataclass
class RAGService:
    chain: Any

    async def ask(
            self,
            question: str,
            history: Optional[list] = None,
            doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            inp = {"question": question, "history": history or [], "doc_id": doc_id}
            out = await self.chain.ainvoke(inp)
            return out.model_dump()
        except OutputParserException as e:
            raw = getattr(e, "llm_output", "") or str(e)
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            return {"question": question, "answer": raw.strip(), "sources": [], "confidence": 0.0,
                    "history": history or []}
        except Exception as e:
            log.exception("RAG ask failed: %s", e)
            return {"question": question, "answer": f"ERROR: {e}", "sources": [], "confidence": 0.0,
                    "history": history or []}


_service: Optional[RAGService] = None


def _resolve_doc_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path
    project_root = Path(__file__).resolve().parents[2]
    candidates = [Path.cwd() / path, project_root / path, project_root / "app" / path]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"PDF not found: DOC_PATH='{raw_path}', tried: {candidates}")


def _build_service_sync() -> RAGService:
    store = PgVectorStore(
        dsn=settings.DATABASE_URL,
        table=settings.PG_TABLE,
        embed_model=settings.EMBED_MODEL,
    )

    if settings.REBUILD_INDEX:
        for doc_cfg in settings.DOCS:
            doc_id = doc_cfg["doc_id"]
            path = _resolve_doc_path(doc_cfg["path"])

            log.warning("Rebuilding index for doc_id=%s", doc_id)
            store.delete_doc(doc_id)

            docs = load_pdf(str(path))
            chunks = split_documents(docs, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            log.info("Chunks created for %s: %d", doc_id, len(chunks))

            store.upsert_chunks(doc_id, chunks)
    else:
        log.info("REBUILD_INDEX=False -> using existing pgvector index (no rebuild)")

    chain = build_chain(
        store=store,
        llm_model=settings.LLM_MODEL,
        k=settings.K_RETRIEVE,
        metric=settings.METRIC,
    )
    return RAGService(chain=chain)


async def init_service() -> RAGService:
    global _service
    if _service is not None:
        return _service
    _service = await anyio.to_thread.run_sync(_build_service_sync)
    return _service


async def get_service() -> RAGService:
    if _service is None:
        return await init_service()
    return _service
