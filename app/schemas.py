from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class Source(BaseModel):
    snippet: str = ""
    meta: dict = Field(default_factory=dict)

class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source] = Field(default_factory=list)
    confidence: float = 0.5
    history: List[dict] = Field(default_factory=list)

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class AskRequest(BaseModel):
    question: str
    history: List[Turn] = Field(default_factory=list)
    doc_id: Optional[str] = None