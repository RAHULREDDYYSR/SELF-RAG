"""
Gradio Web Interface for Self-RAG LangGraph Application

This module provides an interactive web interface using Gradio to interact with
the Self-Reflective Retrieval-Augmented Generation (Self-RAG) workflow.
"""

import gradio as gr
from dotenv import load_dotenv
from graph.graph import app

# Load environment variables
load_dotenv()


def format_documents(documents):
    """Format documents for display in the UI."""
    if not documents:
        return "No documents retrieved."
    
    formatted = []
    for i, doc in enumerate(documents, 1):
        doc_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        formatted.append(f"**Document {i}:**\n{doc_content[:500]}{'...' if len(doc_content) > 500 else ''}")
    
    return "\n\n---\n\n".join(formatted)


def process_question(message, history):
    """
    Process user question through the Self-RAG workflow.
    
    Args:
        message: User's question
        history: Chat history (not used but required by ChatInterface)
    
    Returns:
        Response message with workflow details
    """
    try:
        # Invoke the Self-RAG graph
        result = app.invoke(input={"question": message})
        
        # Extract information from result
        generation = result.get("generation", "No answer generated.")
        documents = result.get("documents", [])
        web_search = result.get("web_search", False)
        
        # Build response message
        response = f"**Answer:**\n{generation}\n\n"
        
        # Add workflow information
        response += "---\n\n**Workflow Information:**\n"
        response += f"- Web Search Triggered: {'Yes ✅' if web_search else 'No ❌'}\n"
        response += f"- Documents Retrieved: {len(documents)}\n\n"
        
        # Add retrieved documents
        if documents:
            response += "---\n\n**Retrieved Documents:**\n\n"
            response += format_documents(documents)
        
        return response
        
    except Exception as e:
        return f"❌ **Error:** {str(e)}\n\nPlease check your environment variables and try again."


# Create Gradio interface  
with gr.Blocks(title="Self-RAG Assistant") as demo:
    gr.Markdown(
        """
        # 🧠 Self-RAG Assistant
        
        **Self-Reflective Retrieval-Augmented Generation** powered by LangGraph
        
        Ask any question and see the Self-RAG workflow in action:
        - 📚 Retrieve relevant documents from ChromaDB
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
        
        1. **Retrieve**: Fetches relevant documents from the vector database
        2. **Grade**: Evaluates document relevance using LLM-based grading
        3. **Decision**: Routes to web search if documents are irrelevant, or directly to generation
        4. **Generate**: Creates a final answer based on the most relevant information
        
        ### ⚙️ Powered by:
        - **LangGraph** for workflow orchestration
        - **ChromaDB** for vector storage
        - **OpenAI GPT-4** for generation
        - **Google Gemini** for document grading
        - **Tavily API** for web search fallback
        """
    )


if __name__ == "__main__":
    print("🚀 Starting Self-RAG Gradio Interface...")
    print("📊 Loading LangGraph workflow...")
    
    # Launch the Gradio app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,  # Automatically open in browser
    )
