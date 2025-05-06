from typing import Any, Dict
from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState)-> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    if any document is not relevant, we will set a flag to run web search
    Args:
        state (dict): the current graph state

    :return:
        state (dict): filtered out irrelevant documents and updated web_search state
    """
    print("🔍 CHECK DOCUMENT RELEVANCE TO QUESTION...")
    question = state['question']
    documents = state['documents']

    filtered_docs =[]
    web_search = False
    for i,doc in enumerate(documents):
        score = retrieval_grader.invoke(
            {'question': question, "document": doc.page_content}
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
    return {"document":filtered_docs, "question":question, "web_search": web_search}
