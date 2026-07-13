from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Load PDF
loader = PyPDFLoader("notes/computer_networks.pdf")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)

# Create vector database
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db"
)

retriever = vector_store.as_retriever()

results = retriever.invoke(
    "explain TCP?"
)

for doc in results:
    print("=" * 50)
    print(doc.page_content)
