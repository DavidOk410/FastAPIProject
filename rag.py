from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
import json, re

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


# ---------- Schema ----------
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


# ---------- Load PDF ----------
doc_path = r".\Fluid-Mechanics-Module-7.pdf"
loader = PyPDFLoader(doc_path)
data = loader.load()
print("done loading....")


# ---------- Split Text into Chunks ----------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print("done splitting....")


# ---------- Vector DB ----------

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag",
)
print("done adding to vector database....")


# ---------- Two LLMs ----------
# LLM для генерации поисковых запросов
llm_query = ChatOllama(model="llama3.1", temperature=0)

# LLM для финального ответа (включить JSON mode)
try:
    llm_answer = ChatOllama(model="llama3.1", temperature=0, format="json")
except TypeError:
    llm_answer = ChatOllama(model="llama3.1", temperature=0)


# ---------- MultiQuery Retriever ----------
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "Generate 5 alternative search queries (one per line) to retrieve relevant documents.\n"
        "Original question: {question}"
    ),
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(search_kwargs={"k": 6}),
    llm_query,
    prompt=QUERY_PROMPT,
)


# ---------- RAG Prompt ----------
rag_template = """Return ONLY valid JSON. No markdown. No extra text.
Follow this schema exactly:

{format_instructions}

Use ONLY the context. If not enough info, set answer to "NOT_FOUND".

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(rag_template)


# ---------- Chain ----------
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


# ---------- Safe invoke ----------
def safe_invoke(question: str) -> dict:
    try:
        out = chain.invoke(question)  # RAGResponse
        return out.model_dump()
    except OutputParserException as e:
        raw = getattr(e, "llm_output", "") or str(e)

        # Попытка вытащить JSON, если он был внутри текста
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        # Жёсткий fallback: всегда валидный JSON
        return {
            "question": question,
            "answer": raw.strip(),
            "sources": [],
            "confidence": 0.0,
        }


# ---------- Test ----------
if __name__ == "__main__":
    res = safe_invoke("what is the document about?")
    print(json.dumps(res, ensure_ascii=False, indent=2))
