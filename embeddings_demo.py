from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector = embeddings.embed_query(
    "The OSI model has seven layers."
)

print(type(vector))
print(len(vector))
print(vector[:10])