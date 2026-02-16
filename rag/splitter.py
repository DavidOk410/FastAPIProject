import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger("rag.splitter")

def split_documents(docs, chunk_size: int, chunk_overlap: int):
    log.info("Splitting docs: chunk_size=%d overlap=%d", chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    log.info("Created %d chunks", len(chunks))
    return chunks
