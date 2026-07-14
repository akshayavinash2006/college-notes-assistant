from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = Chroma(
    persist_directory="db",
    embedding_function=embeddings,
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)