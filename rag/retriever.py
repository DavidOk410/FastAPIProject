from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

def make_retriever(vector_db, llm_query, k: int, n_queries: int = 5):
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            f"Generate {n_queries} alternative search queries (one per line) "
            "to retrieve relevant documents.\n"
            "Original question: {question}"
        ),
    )

    return MultiQueryRetriever.from_llm(
        vector_db.as_retriever(search_kwargs={"k": k}),
        llm_query,
        prompt=query_prompt,
    )
