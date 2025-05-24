from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

llm = init_chat_model("llama3-8b-8192", model_provider="groq")
# llm = init_chat_model("gpt-4o-mini", model_provider="openai")
 
examples = [
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "5"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)


final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# Set the cache for the LLM to InMemoryCache
cache = InMemoryCache()
set_llm_cache(cache)


# pipe symbol gets output of final_prompt and passes it to the LLM
# The LLM will be called with the output of final_prompt as input
chain = final_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

response = chain.invoke({"input": "What's 3 🦜 3?"})
print(response.content)