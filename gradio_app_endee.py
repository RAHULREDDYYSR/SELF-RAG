import gradio as gr
from dotenv import load_dotenv

load_dotenv()

import os
from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import retrieval_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.state import GraphState

from langgraph.graph import END, StateGraph
from langchain_openai import OpenAIEmbeddings
from langchain_endee import EndeeVectorStore
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

base_url = os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1")

vector_store = EndeeVectorStore(
    index_name="rag_endee",
    embedding=OpenAIEmbeddings(),
    dimension=1536,
    space_type="cosine",
    precision="int8",
    base_url=base_url,
)
endee_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

web_search_tool = TavilySearch(max_result=3)


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("⬇️ Retrieving documents from Endee...")
    question = state["question"]
    documents = endee_retriever.invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state: GraphState) -> Dict[str, Any]:
    print("🔍 CHECK DOCUMENT RELEVANCE TO QUESTION...")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for i, doc in enumerate(documents):
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print(f"GRADE: ✅ DOCUMENT({i+1}) RELEVANT ")
            filtered_docs.append(doc)
        else:
            print("GRADE: ❌ DOCUMENT NOT RELEVANT ")
            web_search = True
            continue
    print("▶️ FINISHED CHECKING DOCUMENT RELEVANCE TO QUESTION...")
    return {"document": filtered_docs, "question": question, "web_search": web_search}


def generate(state: GraphState) -> Dict[str, Any]:
    print("🤖 Generating...")
    question = state["question"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0) + 1
    if retry_count >= 3:
        generation = "I could not find a reliable answer in the uploaded documents for this question."
    else:
        generation = generation_chain.invoke({"question": question, "context": documents})
    return {"documents": documents, "question": question, "generation": generation, "retry_count": retry_count}


def web_search(state: GraphState) -> Dict[str, Any]:
    print("🔍 Searching web for relevant documents...")
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})["results"]
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]


def decide_to_generate(state):
    print("🔍 ASSESS GRADED DOCUMENTS...")

    if state["web_search"]:
        print(
            "DECISION: ⭕ NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB_SEARCH"
        )
        return WEBSEARCH
    else:
        print("🤖 DECISION: GENERATE...")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    retry_count = state.get("retry_count", 0)
    if retry_count >= 3:
        print("⛔ MAX RETRIES REACHED. FORCING END.")
        return "useful"

    print("🔍 CHECK HALLUCINATION...")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score
    if grade.lower() == "yes":
        print("✅ DECISION: GENERATION IS GROUNDED IN DOCUMENTS")
        print("🔍 GRADE GENERATION VS QUESTION...")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade.lower() == "yes":
            print("✅ GRADE: GENERATION IS ANSWER TO QUESTION")
            return "useful"
        else:
            print("⭕ GRADE: GENERATION IS NOT ANSWER TO QUESTION")
            return "not useful"
    else:
        print("⭕ DECISION : GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY...")
        return "not supported"


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {"useful": END, "not useful": WEBSEARCH, "not supported": GENERATE},
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

workflow.set_entry_point(RETRIEVE)

app = workflow.compile()


def format_documents(documents):
    if not documents:
        return "No documents retrieved."

    formatted = []
    for i, doc in enumerate(documents, 1):
        doc_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        formatted.append(f"**Document {i}:**\n{doc_content[:500]}{'...' if len(doc_content) > 500 else ''}")

    return "\n\n---\n\n".join(formatted)


def process_question(message, history):
    try:
        result = app.invoke(input={"question": message, "retry_count": 0})

        generation = result.get("generation", "No answer generated.")
        documents = result.get("documents", [])
        web_search_triggered = result.get("web_search", False)

        response = f"**Answer:**\n{generation}\n\n"

        response += "---\n\n**Workflow Information:**\n"
        response += f"- Web Search Triggered: {'Yes ✅' if web_search_triggered else 'No ❌'}\n"
        response += f"- Documents Retrieved: {len(documents)}\n\n"

        if documents:
            response += "---\n\n**Retrieved Documents:**\n\n"
            response += format_documents(documents)

        return response

    except Exception as e:
        return f"❌ **Error:** {str(e)}\n\nPlease check that the Endee server is running and try again."


with gr.Blocks(title="Self-RAG Assistant (Endee)") as demo:
    gr.Markdown(
        """
        # 🧠 Self-RAG Assistant — Endee Edition

        **Self-Reflective Retrieval-Augmented Generation** powered by LangGraph + Endee Vector DB

        Ask any question and see the Self-RAG workflow in action:
        - 📚 Retrieve relevant documents from **Endee** vector database
        - ✅ Grade documents for relevance
        - 🔍 Perform web search if needed
        - 💡 Generate high-quality answers
        """
    )

    chatbot = gr.Chatbot(
        height=500,
        show_label=False,
        avatar_images=(None, "🤖"),
    )

    chat_interface = gr.ChatInterface(
        fn=process_question,
        chatbot=chatbot,
        examples=[
            "What is LCEL?",
            "What is agent memory?",
            "Explain retrieval-augmented generation",
            "What are the benefits of LangGraph?",
            "How does self-reflection improve RAG systems?",
        ],
        cache_examples=False,
    )

    gr.Markdown(
        """
        ---

        ### 📖 How it works:

        1. **Retrieve**: Fetches relevant documents from **Endee** vector database
        2. **Grade**: Evaluates document relevance using LLM-based grading
        3. **Decision**: Routes to web search if documents are irrelevant, or directly to generation
        4. **Generate**: Creates a final answer based on the most relevant information

        ### ⚙️ Powered by:
        - **LangGraph** for workflow orchestration
        - **Endee** for vector storage
        - **OpenAI GPT-4o-mini** for generation
        - **Tavily API** for web search fallback
        """
    )


if __name__ == "__main__":
    print("🚀 Starting Self-RAG Gradio Interface (Endee)...")
    print("📊 Loading LangGraph workflow...")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        inbrowser=True,
    )
