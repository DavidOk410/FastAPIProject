import os
import stat
import time
import shutil
from pathlib import Path
import logging

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

log = logging.getLogger("rag.vectordb")


def _on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def wipe_persist_dir(persist_dir: str):
    p = Path(persist_dir)
    if not p.exists():
        log.warning("Persist dir not found: %s", persist_dir)
        return

    log.warning("Wiping persist dir: %s", persist_dir)
    for attempt in range(5):
        try:
            shutil.rmtree(p, onerror=_on_rm_error)
            log.warning("Wipe success: %s", persist_dir)
            return
        except Exception as e:
            log.warning("Wipe attempt %d failed: %s", attempt + 1, e)
            time.sleep(0.3)

    raise RuntimeError(f"Failed to delete persist dir after retries: {persist_dir}")


def build_chroma_from_documents(chunks, embed_model: str, collection_name: str, persist_dir: str):
    log.info("Building Chroma: collection=%s persist_dir=%s chunks=%d",
             collection_name, persist_dir, len(chunks))
    embeddings = OllamaEmbeddings(model=embed_model)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    try:
        log.info("Chroma count after build: %s", db._collection.count())
    except Exception:
        pass
    return db


def load_chroma(embed_model: str, collection_name: str, persist_dir: str):
    log.info("Loading Chroma: collection=%s persist_dir=%s", collection_name, persist_dir)
    embeddings = OllamaEmbeddings(model=embed_model)
    db = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    try:
        log.info("Chroma count after load: %s", db._collection.count())
    except Exception:
        pass
    return db
