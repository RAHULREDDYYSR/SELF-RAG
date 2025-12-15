from typing import Any, Dict
from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState
import asyncio

async def grade_single_document(question: str, doc: Any) -> tuple[Any, str]:
    """
    Grade a single document asynchronously.
    
    Args:
        question: The user question
        doc: Document to grade
        
    Returns:
        Tuple of (document, grade)
    """
    score = await retrieval_grader.ainvoke(
        {"question": question, "document": doc.page_content}
    )
    grade = score.binary_score
    return doc, grade
    

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    if any document is not relevant, we will set a flag to run web search
    Args:
        state (dict): the current graph state

    :return:
        state (dict): filtered out irrelevant documents and updated web_search state
    """
    print("🔍 CHECK DOCUMENT RELEVANCE TO QUESTION...")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    
    # Grade all doc parallel
    async def grade_all():
        tasks = [grade_single_document(question=question,doc=doc) for doc in documents]
        return await asyncio.gather(*tasks)
    
    # run the async grading
    results = asyncio.run(grade_all())

    for doc, grade in results:
        if grade.lower() == 'yes':
            print("✅ Document is relevant to the question")
            filtered_docs.append(doc)
        else:
            print("❌ Document is not relevant to the question")
            web_search = True
    return {"documents": filtered_docs, "web_search": web_search}
