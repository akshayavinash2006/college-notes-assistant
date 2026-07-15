import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

print("1. Testing Embeddings ONLY:")
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector = embeddings.embed_query("What is the Transmission Control Protocol (TCP)?")
    print(f"   Success! Vector length: {len(vector)}")
except Exception as e:
    print(f"   Failed: {e}")

print("\n2. Initializing ChatGoogleGenerativeAI:")
try:
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
    print("   LLM Initialized.")
except Exception as e:
    print(f"   Failed LLM Init: {e}")

print("\n3. Testing Embeddings AFTER LLM:")
try:
    embeddings2 = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector2 = embeddings2.embed_query("What is the Transmission Control Protocol (TCP)?")
    print(f"   Success! Vector length: {len(vector2)}")
except Exception as e:
    print(f"   Failed: {e}")
