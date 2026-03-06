# app/main.py
from contextlib import asynccontextmanager
from pathlib import Path
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import AskRequest
from app.rag.service import init_service, get_service
from app.logging_conf import setup_logging

setup_logging()
log = logging.getLogger("app")

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_FILE = BASE_DIR / "index.html"


@asynccontextmanager
async def lifespan(app_: FastAPI):
    await init_service()
    yield


app = FastAPI(
    title="Async PDF RAG API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files under /static
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

@app.get("/docs-list")
async def docs_list():
    return [
        {"doc_id": "fluid-mechanics-module-7", "path": "Fluid-Mechanics-Module-7.pdf"},
        {"doc_id": "python-crash-course", "path": "Python Crash Course.pdf"},
    ]


@app.get("/")
async def root_ui():
    return FileResponse(INDEX_FILE)


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/ask")
async def ask(req: AskRequest):
    service = await get_service()
    history = [t.model_dump() if hasattr(t, "model_dump") else t for t in (req.history or [])]

    return await service.ask(
        req.question,
        history=history,
        doc_id=req.doc_id,
    )