from langchain.prompts import ChatPromptTemplate

def make_rag_prompt(format_instructions: str):
    rag_template = """Return ONLY valid JSON. No markdown. No extra text.
Follow this schema exactly:

{format_instructions}

Use ONLY the context. If not enough info, set answer to "NOT_FOUND".

Context:
{context}

Question: {question}
"""
    return ChatPromptTemplate.from_template(rag_template)
