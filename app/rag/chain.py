# app/rag/chain.py
import json
import logging
import re
import time
from typing import Any, Dict, Union

import anyio
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from app.rag.pgvector import PgVectorStore
from app.schemas import RAGResponse, Source  # <-- unified schemas

log = logging.getLogger("rag.chain")


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
    - Base the answer primarily on the context.
    - If the context is incomplete, say so clearly.
    - State the reasons of low confidence in the end of the answer. 
    - Do not sound overly certain unless the context is explicit.
    - If you are printing code, print it on different lines.
    - If not enough info is available, answer with "NOT_FOUND".
    
    Conversation history:
    {history}
    
    Context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(rag_template)

    try:
        llm = ChatOllama(model=llm_model, temperature=0, base_url="http://ollama:11434", format="json")
    except TypeError:
        llm = ChatOllama(model=llm_model, temperature=0, base_url="http://ollama:11434")

    def sanitize_sources(data: dict) -> dict:
        srcs = data.get("sources", [])
        clean = []
        if isinstance(srcs, list):
            for s in srcs:
                if isinstance(s, dict):
                    clean.append(
                        {
                            "snippet": s.get("snippet", "") or "",
                            "meta": s.get("meta", {}) or {},
                        }
                    )
        data["sources"] = clean
        return data

    async def ainvoke(inp: Union[str, Dict[str, Any]]) -> RAGResponse:
        # 1) Parse input
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

        # 2) Retrieval with timeout
        t0 = time.time()
        try:
            with anyio.fail_after(30):
                docs = store.similarity_search(
                    query=question,
                    k=k,
                    metric=metric,
                    filters=filters,
                )
        except TimeoutError:
            return RAGResponse(
                question=question,
                answer="ERROR: retrieval timeout (DB/embeddings)",
                sources=[],
                confidence=0.0,
                history=history,
            )
        log.info("retrieval took %.2fs", time.time() - t0)

        context = "\n\n".join(d.page_content for d in docs)
        history_text = format_history(history)

        msg = prompt.format_messages(
            history=history_text,
            context=context,
            question=question,
        )

        # 3) LLM with timeout
        t1 = time.time()
        try:
            with anyio.fail_after(60):
                raw = await llm.ainvoke(msg)
        except TimeoutError:
            return RAGResponse(
                question=question,
                answer="ERROR: llm timeout (Ollama chat)",
                sources=[],
                confidence=0.0,
                history=history,
            )
        log.info("llm took %.2fs", time.time() - t1)

        text = getattr(raw, "content", raw)

        # 4) Parse JSON response
        try:
            resp = parser.parse(text)
            resp.history = history  # force echo
            return resp
        except OutputParserException:
            m = re.search(r"\{.*\}", str(text), flags=re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                data = sanitize_sources(data)
                data["history"] = history
                return RAGResponse(**data)

            return RAGResponse(
                question=question,
                answer=str(text),
                sources=[],
                confidence=0.0,
                history=history,
            )

    class _Chain:
        async def ainvoke(self, inp: Union[str, Dict[str, Any]]):
            return await ainvoke(inp)

    return _Chain()
