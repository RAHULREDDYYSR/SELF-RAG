from dotenv import load_dotenv
from sympy.codegen import Print

load_dotenv()
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader

from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import  GraphState


def decide_to_generate(state):
    print("🔍 ASSESS GRADED DOCUMENTS...")

    if state['web_search']:
        print(
            "DECISION: ⭕ NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB_SEARCH"
        )
        return WEBSEARCH
    else:
        print("🤖 DECISION: GENERATE...")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState)->  str:
    print("🔍 CHECK HALLUCINATION...")
    question = state['question']
    documents = state['documents']
    generation = state['generation']

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score
    if grade.lower() == "yes":
        print("✅ DECISION: GENERATION IS GROUNDED IN DOCUMENTS")
        print("🔍 GRADE GENERATION VS QUESTION...")
        score = answer_grader.invoke(
            {"question": question, "generation": generation}
        )
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



# add all the nodes
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# add edges
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,  # condition node
    decide_to_generate,  # condition
            {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE
    }
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "useful": END,
        "not useful": WEBSEARCH,
        "not supported": GENERATE
    }
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

# starting point
workflow.set_entry_point(RETRIEVE)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")