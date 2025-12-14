from dotenv import load_dotenv

load_dotenv()
from graph.chains.hallucination_grader import GradeHallucination


from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucination_grader


def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrival_grade_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pancakes", "document": doc_text}
    )
    assert res.binary_score == "no"


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})
    res: GradeHallucination = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score == "yes"


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})
    res: GradeHallucination = hallucination_grader.invoke(
        {"documents": docs, "generation": "this is a test"}
    )
    assert res.binary_score == "no"
