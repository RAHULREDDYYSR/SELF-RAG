from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence




class GradeAnswer(BaseModel):
    """Binary score for correctness of answer."""
    binary_score: str = Field(description="Answer is correct, 'yes' or 'no'")


llm = ChatOpenAI(model='gpt-4o-mini')
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """you are a grader assessing whether an answer addresses / resolves a question. \n
     If the answer is correct, grade it as correct. \n
     Give a binary score 'yes' or 'no' score to indicate whether the answer is correct."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Answer: \n\n {generation} \n\n Question: {question}"),
    ]
)
answer_grader: RunnableSequence = answer_prompt | structured_llm_grader