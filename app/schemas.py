from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import PydanticOutputParser

class Source(BaseModel):
    page: int = Field(..., description="Page number in the PDF (1-based)")
    snippet: str = Field(..., description="Short quote from the context")

class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source] = Field(default_factory=list)
    confidence: float = 0.5

parser = PydanticOutputParser(pydantic_object=RAGResponse)
format_instructions = parser.get_format_instructions()

# Для API (запрос/ответ)
class AskRequest(BaseModel):
    question: str
