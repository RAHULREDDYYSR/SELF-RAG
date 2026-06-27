from dotenv import load_dotenv

load_dotenv()
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_endee import EndeeVectorStore

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

base_url = os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1")

vector_store = EndeeVectorStore(
    index_name="rag_endee",
    embedding=OpenAIEmbeddings(),
    dimension=1536,
    space_type="cosine",
    precision="int8",
    base_url=base_url,
)

for doc in doc_splits:
    doc.metadata = {"source": doc.metadata.get("source", "")}

vector_store.add_documents(doc_splits)
print("Done! Ingested documents into Endee index 'rag_endee'.")
