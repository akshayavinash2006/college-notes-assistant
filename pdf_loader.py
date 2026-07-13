from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("notes/computer_networks.pdf")

docs = loader.load()

print(docs)