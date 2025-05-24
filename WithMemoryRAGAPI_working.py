from  dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import asyncio
from contextlib import asynccontextmanager
import bs4

# Community imports to avoid deprecation warnings
from langchain_community.chat_models import ChatOpenAI

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# Core LangChain utilities
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model

# Globals for vector store, retriever, and chain
embeddings: OpenAIEmbeddings
vectordb: Chroma
history_retriever: Any
rag_chain: Any
memories: Dict[str, ConversationBufferMemory] = {}

# Request schema
class QueryRequest(BaseModel):
    user_id: str
    query: str

# Lifespan handler for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup: load and index documents ---
    loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
        ),
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Initialize embeddings and Chroma
    global embeddings, vectordb, history_retriever, rag_chain
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma")

    # Base retriever and LLM
    base_retriever = vectordb.as_retriever(search_kwargs={"k": 1})
    # llm = ChatOpenAI(model_name="gpt-3.5", temperature=0.0)
    llm = init_chat_model("llama3-8b-8192", model_provider="groq")

    # History-aware retriever
    search_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "Generate a precise search query for retrieval.")
    ])
    history_retriever = create_history_aware_retriever(
        llm=llm, retriever=base_retriever, prompt=search_prompt
    )

    # Document-combining chain
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the context below to answer the question:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=answer_prompt)

    # Final RAG chain
    rag_chain = create_retrieval_chain(retriever=history_retriever, combine_docs_chain=doc_chain)
    

    yield  # application is now running
    # --- shutdown (optional cleanup) ---

# FastAPI app setup with lifespan
app = FastAPI(
    title="RAG Chat Service",
    description="A RAG-based conversational API using LangChain, Chroma DB, and FastAPI.",
    lifespan=lifespan
)

@app.post("/chat")
async def chat(request: QueryRequest) -> Any:
    user_id, query = request.user_id, request.query

    # Initialize per-user memory
    if user_id not in memories:
        memories[user_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = memories[user_id]

    # Retrieve history
    hist_vars = memory.load_memory_variables({})
    history: List[Any] = hist_vars.get("chat_history", [])

    try:
        # Run RAG chain in thread to avoid blocking
        # outputs = await asyncio.to_thread(rag_chain, {"input": query, "chat_history": history})
        outputs = await rag_chain.ainvoke({"input": query, "chat_history": history})

        answer = outputs.get("answer")

        # Persist conversation
        memory.save_context({"input": query}, {"output": answer})
        stored = memory.load_memory_variables({}).get("chat_history", [])

        return {"answer": answer, "chat_history": [m.content for m in stored]}
    except Exception as e:
        # Log the error (optional)
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# To run: uvicorn simpleRAGAPI:app --host 0.0.0.0 --port 8000