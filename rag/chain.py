from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough

from app.schemas import parser, format_instructions
from app.rag.prompts import make_rag_prompt
from app.rag.retriever import make_retriever

def build_chain(vector_db, llm_model: str, k: int, n_queries: int):
    # LLM для query generation
    llm_query = ChatOllama(model=llm_model, temperature=0)

    # LLM для финального ответа (пытаемся включить json mode)
    try:
        llm_answer = ChatOllama(model=llm_model, temperature=0, format="json")
    except TypeError:
        llm_answer = ChatOllama(model=llm_model, temperature=0)

    retriever = make_retriever(vector_db, llm_query, k=k, n_queries=n_queries)
    prompt = make_rag_prompt(format_instructions)

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "format_instructions": lambda _: format_instructions,
        }
        | prompt
        | llm_answer
        | parser
    )
    return chain
