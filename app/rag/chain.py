import re
import json
from typing import Any, Dict, Union

from pydantic import BaseModel, Field
from typing import List as TList

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

from app.rag.pgvector import PgVectorStore


class Source(BaseModel):
    snippet: str = ""  # forgiving default
    meta: dict = Field(default_factory=dict)


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: TList[Source] = Field(default_factory=list)
    confidence: float = 0.5
    history: list = Field(default_factory=list)  # echo input history back


def build_chain(store: PgVectorStore, llm_model: str, k: int, metric: str):
    parser = PydanticOutputParser(pydantic_object=RAGResponse)

    def format_history(history: list) -> str:
        lines = []
        for t in history or []:
            role = t.get("role")
            content = t.get("content", "")
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    # Avoid passing pydantic schema refs ($ref) to the model; give a simple explicit format instead.
    rag_template = """Return ONLY valid JSON (no markdown, no extra text).

JSON format:
{{
  "question": "...",
  "answer": "...",
  "sources": [{{"snippet": "...", "meta": {{}}}}],
  "confidence": 0.0,
  "history": []
}}

Rules:
- Use ONLY the context.
- If not enough info, set answer to "NOT_FOUND".
- sources must be a list of objects with keys "snippet" and "meta".
- history in the response must be EXACTLY the same list you received in the input (echo it back unchanged).

Conversation history (for interpreting follow-ups, not a source of truth):
{history}

Context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(rag_template)

    try:
        llm = ChatOllama(model=llm_model, temperature=0, format="json")
    except TypeError:
        llm = ChatOllama(model=llm_model, temperature=0)

    def _sanitize_sources(data: dict) -> dict:
        srcs = data.get("sources", [])
        clean = []
        for s in srcs if isinstance(srcs, list) else []:
            if isinstance(s, dict):
                clean.append({"snippet": s.get("snippet", ""), "meta": s.get("meta", {}) or {}})
        data["sources"] = clean
        return data

    async def ainvoke(inp: Union[str, Dict[str, Any]]) -> RAGResponse:
        # Accept either plain question string OR dict {"question": ..., "history": ..., "doc_id": ...}
        if isinstance(inp, str):
            question = inp
            history = []
            doc_id_inp = None
        elif isinstance(inp, dict):
            question = inp.get("question", "")
            history = inp.get("history", []) or []
            doc_id_inp = inp.get("doc_id")
        else:
            raise TypeError(f"Invalid chain input type: {type(inp)}")

        if not isinstance(question, str):
            raise TypeError(f"'question' must be a string, got: {type(question)}")

        filters = {"doc_id": doc_id_inp} if doc_id_inp else None

        docs = store.similarity_search(
            query=question,
            k=k,
            metric=metric,
            filters=filters,
        )

        context = "\n\n".join(d.page_content for d in docs)
        history_text = format_history(history)

        msg = prompt.format_messages(
            history=history_text,
            context=context,
            question=question,
        )

        raw = await llm.ainvoke(msg)
        text = getattr(raw, "content", raw)

        try:
            resp = parser.parse(text)
            # force echo history back unchanged (regardless of what the model returned)
            resp.history = history
            return resp
        except OutputParserException:
            m = re.search(r"\{.*\}", str(text), flags=re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                data = _sanitize_sources(data)
                data["history"] = history  # force echo
                return RAGResponse(**data)

            return RAGResponse(
                question=question,
                answer=str(text),
                sources=[],
                confidence=0.0,
                history=history,  # force echo
            )

    class _Chain:
        async def ainvoke(self, inp: Union[str, Dict[str, Any]]):
            return await ainvoke(inp)

    return _Chain()