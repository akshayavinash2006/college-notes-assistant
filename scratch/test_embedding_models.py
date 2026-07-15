import sys
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

for model in ["models/gemini-embedding-001", "models/embedding-001", "models/text-embedding-004"]:
    print(f"Testing model: {model}")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=model)
        vector = embeddings.embed_query("What is the Transmission Control Protocol (TCP)?")
        print(f"  Success! Vector length: {len(vector)}")
    except Exception as e:
        print(f"  Failed: {e}")
