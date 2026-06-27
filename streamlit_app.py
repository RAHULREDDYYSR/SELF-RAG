import streamlit as st
import os
import tempfile

from dotenv import load_dotenv
load_dotenv()

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
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from endee import Endee

INDEX_NAME = "rag_streamlit"
EMBEDDING_DIM = 1536
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
base_url = os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1")

if "ready" not in st.session_state:
    st.session_state.ready = False
if "app" not in st.session_state:
    st.session_state.app = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []
if "ingested_chunks" not in st.session_state:
    st.session_state.ingested_chunks = 0


def clear_index():
    client = Endee()
    client.set_base_url(base_url)
    try:
        client.delete_index(INDEX_NAME)
        print(f"🗑️ Deleted index '{INDEX_NAME}'")
    except Exception as e:
        print(f"Note: Could not delete index ({e})")


def verify_retrieval(retriever, test_query="what is this document about?"):
    print(f"🔍 VERIFYING retrieval with query: '{test_query}'")
    test_results = retriever.invoke(test_query)
    print(f"   Retrieved {len(test_results)} docs in verification")
    for r in test_results:
        print(f"   - {r.page_content[:60]}...")
    return test_results


def build_graph(retriever):
    web_search_tool = TavilySearch(max_result=3)

    def retrieve(state: GraphState) -> Dict[str, Any]:
        print("⬇️ Retrieving documents from Endee...")
        question = state["question"]
        documents = retriever.invoke(question)
        print(f"   Retrieved {len(documents)} docs")
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
            print("DECISION: ⭕ NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB_SEARCH")
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
    return workflow.compile()


def ingest_files(uploaded_files):
    print(f"📥 Starting ingestion of {len(uploaded_files)} file(s)...")
    all_docs = []
    for f in uploaded_files:
        print(f"  Processing: {f.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f.name) as tmp:
            tmp.write(f.getvalue())
            path = tmp.name
        try:
            if f.name.lower().endswith(".pdf"):
                print(f"  -> Loading as PDF...")
                loader = PyPDFLoader(path)
            elif f.name.lower().endswith(".docx"):
                print(f"  -> Loading as DOCX...")
                loader = Docx2txtLoader(path)
            else:
                print(f"  -> Skipped unsupported type")
                st.warning(f"Skipped unsupported file: {f.name}")
                continue
            loaded = loader.load()
            print(f"  -> Loaded {len(loaded)} page(s)")
            all_docs.extend(loaded)
        except Exception as e:
            print(f"  ❌ Error loading file: {e}")
            st.error(f"Error loading {f.name}: {e}")
        finally:
            os.unlink(path)

    if not all_docs:
        print("❌ No docs loaded from any file")
        st.error("No documents could be loaded from the uploaded files.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(all_docs)
    print(f"✂️ Split into {len(chunks)} chunks")

    for d in chunks:
        src = d.metadata.get("source", "upload")
        d.metadata = {"source": src[:200]}

    print("🗑️ Clearing old index...")
    clear_index()

    print("🔗 Creating EndeeVectorStore...")
    vs = EndeeVectorStore(
        index_name=INDEX_NAME,
        embedding=OpenAIEmbeddings(),
        dimension=EMBEDDING_DIM,
        space_type="cosine",
        precision="int8",
        base_url=base_url,
    )
    print(f"📤 Adding {len(chunks)} chunks to Endee...")
    vs.add_documents(chunks)
    print("✅ Documents added to Endee")

    retriever = vs.as_retriever(search_kwargs={"k": 4})

    print("🔍 Verifying retrieval works...")
    test_results = verify_retrieval(retriever, "what is this about")
    if len(test_results) == 0:
        print("⚠️ RETRIEVAL VERIFICATION FAILED: 0 documents returned!")
        st.warning("⚠️ Documents were added but retrieval returned 0 results. Chunks may be empty or embeddings may not match.")
    else:
        print(f"✅ VERIFICATION OK: {len(test_results)} docs retrieved")

    print("🤖 Building Self-RAG graph...")
    st.session_state.app = build_graph(retriever)
    st.session_state.ready = True
    st.session_state.ingested_files = [f.name for f in uploaded_files]
    st.session_state.ingested_chunks = len(chunks)
    st.session_state.messages = []
    print("🎉 Ingestion complete!")


st.set_page_config(
    page_title="Self-RAG with Endee",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Self-RAG with Endee")
st.markdown("Upload **PDF** or **DOCX** documents and ask questions about their content.")

with st.sidebar:
    st.header("📄 Document Upload")

    uploaded_files = st.file_uploader(
        "Choose PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files and st.button("🚀 Ingest Documents", type="primary", use_container_width=True):
        with st.spinner("Ingesting documents into Endee..."):
            try:
                ingest_files(uploaded_files)
                st.success(f"Ingested {len(uploaded_files)} file(s) successfully!")
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

    st.divider()

    if st.session_state.ready:
        st.subheader("📁 Ingested Files")
        for name in st.session_state.ingested_files:
            st.write(f"- {name}")
        st.caption(f"📊 {st.session_state.ingested_chunks} text chunks indexed in Endee")

        if st.button("🔄 Clear & Reset", use_container_width=True):
            st.session_state.ready = False
            st.session_state.app = None
            st.session_state.messages = []
            st.session_state.ingested_files = []
            st.session_state.ingested_chunks = 0
            try:
                clear_index()
            except Exception:
                pass
    else:
        st.info("👈 Upload documents to get started.")

    st.divider()
    st.caption(f"**Endee index:** `{INDEX_NAME}`")
    st.caption(f"**Server:** `{base_url}`")

st.divider()

if not st.session_state.ready:
    st.info(
        "📂 **No documents ingested yet.**\n\n"
        "Upload PDF or DOCX files in the sidebar and click **Ingest Documents** to begin."
    )
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "details" in msg:
                with st.expander("📋 Workflow Details"):
                    st.markdown(msg["details"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Running Self-RAG workflow..."):
                try:
                    print(f"\n🔹 USER QUESTION: '{prompt}'")
                    result = st.session_state.app.invoke(input={"question": prompt, "retry_count": 0})

                    generation = result.get("generation", "No answer generated.")
                    documents = result.get("documents", [])
                    web_search_triggered = result.get("web_search", False)
                    print(f"🔹 RESULT: {len(documents)} docs, web_search={web_search_triggered}")
                    print(f"🔹 GENERATION: {generation[:100]}...")

                    answer = f"**Answer:**\n{generation}"
                    st.markdown(answer)

                    details = ""
                    details += f"- **Web Search Triggered:** {'Yes' if web_search_triggered else 'No'}\n"
                    details += f"- **Documents Retrieved:** {len(documents)}\n\n"

                    if documents:
                        details += "**Retrieved Documents:**\n\n"
                        for i, doc in enumerate(documents, 1):
                            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                            details += f"**Document {i}:**\n{content[:400]}{'...' if len(content) > 400 else ''}\n\n---\n\n"

                    with st.expander("📋 Workflow Details"):
                        st.markdown(details)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "details": details,
                    })

                except Exception as e:
                    error_msg = f"❌ **Error:** {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })
