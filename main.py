from dotenv import load_dotenv

load_dotenv()
from langchain_core.output_parsers import StrOutputParser

from graph.graph import app

if __name__ == "__main__":
    print("Self_RAG in work...")
    print(app.invoke(input={"question": "what is lcel?"}))
