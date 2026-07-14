from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Load every PDF inside notes/
documents = []

notes_folder = Path("notes")

for pdf in notes_folder.glob("*.pdf"):
    print(f"Loading {pdf.name}")

    loader = PyPDFLoader(str(pdf))
    docs = loader.load()

    documents.extend(docs)

print(f"\nLoaded {len(documents)} pages.")

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

# Build database
Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db",
)

print("\nVector database created successfully.")