import sys
from vectorstore import retriever

sys.stdout.reconfigure(encoding='utf-8')

query = "What is computer networks?"
docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents:")
for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} (Source: {doc.metadata.get('source')} Page: {doc.metadata.get('page')}) ---")
    print(doc.page_content[:500])
