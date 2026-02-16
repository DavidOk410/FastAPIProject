import shutil
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

def wipe_persist_dir(persist_dir: str):
    p = Path(persist_dir)
    if p.exists() and p.is_dir():
        shutil.rmtree(p)

def build_chroma_from_documents(chunks, embed_model: str, collection_name: str, persist_dir: str):
    embeddings = OllamaEmbeddings(model=embed_model)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )

def load_chroma(embed_model: str, collection_name: str, persist_dir: str):
    embeddings = OllamaEmbeddings(model=embed_model)
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
