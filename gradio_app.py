"""
Gradio Web Interface for C-RAG (Corrective Retrieval-Augmented Generation)

A simple and elegant UI to interact with the C-RAG system.
"""
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from graph.graph import app as c_rag_app



def process_question(question: str, show_details: bool = True, progress=gr.Progress()):
    """
    Process a question through the C-RAG system and return formatted results.
    
    Args:
        question: User's question
        show_details: Whether to show detailed workflow information
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (answer, details, sources)
    """
    if not question.strip():
        return "⚠️ Please enter a question.", "", ""
    
    progress(0.1, desc="Initializing C-RAG...")
    
    # Run the C-RAG graph
    progress(0.3, desc="Running Retrieval & Grading (this may take a moment)...")
    result = c_rag_app.invoke(input={"question": question})
    
    progress(0.9, desc="Formatting Results...")
    
    # Extract information
    answer = result.get("generation", "No answer generated")
    documents = result.get("documents", [])
    web_search_used = result.get("web_search", False)
    
    # Format the main answer
    answer_text = f"### 🤖 Answer\n\n{answer}"
    
    # Format workflow details
    details_text = ""
    if show_details:
        details_parts = []
        
        # Retrieval info
        details_parts.append(f"📚 **Retrieved Documents:** {len(documents)}")
        
        # Web search info
        if web_search_used:
            details_parts.append("🔍 **Web Search:** Used (some documents were not relevant)")
        else:
            details_parts.append("✅ **Web Search:** Not needed (all documents were relevant)")
        
        # Grading summary
        details_parts.append(f"✓ **Relevant Documents:** {len(documents)}")
        
        details_text = "\n\n".join(details_parts)
    
    # Format sources
    sources_text = ""
    if documents:
        sources_parts = ["### 📄 Source Documents\n"]
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            sources_parts.append(f"**{i}. {source}**\n\n{content_preview}\n")
        sources_text = "\n".join(sources_parts)
    
    return answer_text, details_text, sources_text


def create_ui():
    """Create the Gradio interface."""
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="C-RAG - Corrective RAG System",
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        #title {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 0.5em;
        }
        #subtitle {
            text-align: center;
            color: #666;
            font-size: 1.2em;
            margin-bottom: 2em;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
            <div id="title">🧠 C-RAG System</div>
            <div id="subtitle">Corrective Retrieval-Augmented Generation with Intelligent Document Grading</div>
        """)
        
        # Main interface
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Ask a Question",
                    placeholder="e.g., What are the components of an autonomous agent?",
                    lines=3,
                    autofocus=True
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Get Answer", variant="primary", size="lg")
                    clear_btn = gr.ClearButton([question_input], value="🗑️ Clear")
                
                show_details = gr.Checkbox(
                    label="Show workflow details",
                    value=True,
                    info="Display retrieval and grading information"
                )
        
        # Results section
        with gr.Row():
            with gr.Column():
                answer_output = gr.Markdown(label="Answer")
        
        with gr.Row():
            with gr.Column(scale=1):
                details_output = gr.Markdown(label="Workflow Details")
            with gr.Column(scale=2):
                sources_output = gr.Markdown(label="Sources")
        
        # Examples
        gr.Examples(
            examples=[
                ["What are the key components of LLM-powered autonomous agents?"],
                ["Explain Chain of Thought (CoT) prompting."],
                ["What are some common adversarial attacks on LLMs?"],
                ["Describe the ReAct pattern for agents."],
            ],
            inputs=question_input,
            label="💡 Example Questions (Based on Ingested Docs)"
        )
        
        # Footer
        gr.HTML("""
            <div style="text-align: center; margin-top: 2em; padding-top: 1em; border-top: 1px solid #eee; color: #666;">
                <p><strong>How it works:</strong> Retrieves documents → Grades relevance → Corrects with web search if needed → Generates answer</p>
                <p>Built with LangChain, ChromaDB, and LangGraph | Powered by OpenAI & Google Gemini</p>
            </div>
        """)
        
        # Event handlers
        submit_btn.click(
            fn=process_question,
            inputs=[question_input, show_details],
            outputs=[answer_output, details_output, sources_output]
        )
        
        question_input.submit(
            fn=process_question,
            inputs=[question_input, show_details],
            outputs=[answer_output, details_output, sources_output]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True  # Automatically open in browser
    )
