import sys
from dotenv import load_dotenv
from config import llm
from vectorstore import retriever
from rag import history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

# Mock history
chat_history = [
    HumanMessage(content="what are the two main protocols"),
    AIMessage(content="The two main protocols in the transport layer are UDP and TCP.")
]

# Run history_aware_retriever
try:
    print("Invoking history_aware_retriever...")
    res = history_aware_retriever.invoke({
        "input": "what is TCP",
        "chat_history": chat_history
    })
    print(f"Success! Retrieved {len(res)} docs.")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
