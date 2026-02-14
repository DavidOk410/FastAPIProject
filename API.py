from fastapi import FastAPI
from pydantic import BaseModel
from rag import safe_invoke   # import from rag.py

app = FastAPI()

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: AskRequest):
    return safe_invoke(req.question)