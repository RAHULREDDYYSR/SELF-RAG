from typing import Any, Dict
from dotenv import load_dotenv
load_dotenv()
from langchain.schema import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState

web_search_tool = TavilySearch(max_result=3)


def web_search(state: GraphState)-> Dict[str, Any]:
    print("🔍 Searching web for relevant documents...")
    question = state['question']
    documents = state['documents']

    tavily_results = web_search_tool.invoke({"query": question})['results']
    joined_tavily_result = "\n".join(
        [tavily_result['content'] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]