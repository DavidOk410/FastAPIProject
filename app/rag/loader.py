import logging
from langchain_community.document_loaders import PyPDFLoader

log = logging.getLogger("rag.loader")

def load_pdf(path: str):
    log.info("Loading PDF: %s", path)
    docs = PyPDFLoader(path).load()
    log.info("Loaded %d pages", len(docs))
    if docs:
        preview = (docs[0].page_content or "")[:200].replace("\n", " ")
        log.info("Page 0 preview: %s", preview)
    return docs
