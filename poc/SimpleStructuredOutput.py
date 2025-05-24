from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph,MessagesState 
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition




llm = init_chat_model("llama3-8b-8192", model_provider="groq")
# llm = init_chat_model("gpt-4o-mini", model_provider="openai")


json_schema = {
    "title": "joke",
    "description": "Joke to tell user.",
    "type": "object",
    "properties": {
        "setup": {
            "type": "string",
            "description": "The setup of the joke",
        },
        "punchline": {
            "type": "string",
            "description": "The punchline to the joke",
        },
        "rating": {
            "type": "integer",
            "description": "How funny the joke is, from 1 to 10",
            "default": None,
        },
    },
    "required": ["setup", "punchline"],
}

 
structured_llm = llm.with_structured_output(json_schema)


 
system_prompt = SystemMessage(content="You are a helpful assistant that tells jokes about food.")
response = structured_llm.invoke([system_prompt,"Tell me "])
print(response)

## Streaming output
# for chunk in structured_llm.stream([
#     system_prompt,
#     "Tell me a joke "
# ]):
#     print(chunk)