from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.schemas import AskRequest
from app.rag.service import init_service, get_service   # ✅ исправлено

from app.logging_conf import setup_logging
setup_logging()
import logging
logging.getLogger("test").info("Logging works")


@asynccontextmanager
async def lifespan(app_: FastAPI):
    await init_service()
    yield


app = FastAPI(
    title="Async PDF RAG API",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {"status": "ok", "docs": "/docs"}


@app.post("/ask")
async def ask(req: AskRequest):
    service = await get_service()
    return await service.ask(req.question)
