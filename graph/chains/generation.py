from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-nano")
prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm |StrOutputParser()