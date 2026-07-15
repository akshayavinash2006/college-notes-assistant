import streamlit as st
import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from vectorstore import embeddings, vector_store
from rag import conversational_rag_chain

# Create folders if they don't exist
notes_dir = Path("notes")
notes_dir.mkdir(exist_ok=True)

# -----------------------------------------
# Page Configurations & Styling
# -----------------------------------------
st.set_page_config(
    page_title="College Notes Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (Google Font, gradients, dark/light balance, clean bubbles)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title gradient */
    .title-text {
        font-weight: 800;
        background: linear-gradient(90deg, #FF7E5F, #FEB47B, #86A8E7, #91EAE4);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 8s ease infinite;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #F8FAFC;
    }
    
    /* Chat message container custom styles */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    
    /* Source badges */
    .source-badge {
        display: inline-block;
        background: rgba(134, 168, 231, 0.15);
        color: #86A8E7;
        border: 1px solid rgba(134, 168, 231, 0.3);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        margin-right: 6px;
        margin-top: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Session State Initialization
# -----------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    # Use a unique session ID per browser run
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# -----------------------------------------
# Document Ingestion Function
# -----------------------------------------
def run_ingestion():
    with st.spinner("Ingesting and index building..."):
        # Load every PDF inside notes/
        documents = []
        pdf_files = list(notes_dir.glob("*.pdf"))
        
        if not pdf_files:
            st.sidebar.warning("No notes found to ingest.")
            return False

        for pdf in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf))
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                st.sidebar.error(f"Error loading {pdf.name}: {e}")

        if not documents:
            return False

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)

        # Rebuild/Reset database to avoid duplicates
        try:
            # Clear existing documents in vector store
            existing_data = vector_store.get()
            if existing_data and "ids" in existing_data and existing_data["ids"]:
                vector_store.delete(ids=existing_data["ids"])
            
            # Add new chunks
            vector_store.add_documents(chunks)
            st.sidebar.success(f"Successfully indexed {len(pdf_files)} PDFs ({len(chunks)} chunks).")
            return True
        except Exception as e:
            st.sidebar.error(f"Failed to build index: {e}")
            return False

# -----------------------------------------
# Sidebar
# -----------------------------------------
with st.sidebar:
    st.markdown("### 🎓 Notes Assistant Settings")
    st.write("Upload class notes, textbooks, or reference papers to chat with them.")
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload PDF Notes",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        files_saved = False
        for uploaded_file in uploaded_files:
            file_path = notes_dir / uploaded_file.name
            if not file_path.exists():
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                files_saved = True
                st.sidebar.info(f"Saved: {uploaded_file.name}")
        
        if files_saved:
            run_ingestion()
            st.rerun()

    st.markdown("---")
    st.markdown("### 📂 Uploaded Notes")
    existing_pdfs = list(notes_dir.glob("*.pdf"))
    if existing_pdfs:
        for pdf in existing_pdfs:
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {pdf.name}")
            if col2.button("🗑️", key=f"del_{pdf.name}"):
                os.remove(pdf)
                # Re-run ingestion to remove deleted file from vector database
                if list(notes_dir.glob("*.pdf")):
                    run_ingestion()
                else:
                    # If no PDFs left, clear the db
                    existing_data = vector_store.get()
                    if existing_data and "ids" in existing_data and existing_data["ids"]:
                        vector_store.delete(ids=existing_data["ids"])
                st.rerun()
    else:
        st.write("*No documents uploaded yet.*")

    st.markdown("---")
    if st.button("🔄 Force Re-index"):
        run_ingestion()

# -----------------------------------------
# Main UI
# -----------------------------------------
st.markdown('<div class="title-text">College Notes Assistant</div>', unsafe_allow_html=True)
st.write("Ask questions about your uploaded PDFs. The assistant uses ONLY the context from your notes.")

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            sources_html = "".join([f'<span class="source-badge">📖 {src}</span>' for src in message["sources"]])
            st.markdown(f'<div style="margin-top: 10px;">{sources_html}</div>', unsafe_allow_html=True)

# User Chat Input
if question := st.chat_input("Ask a question about your notes..."):
    # Display user message
    with st.chat_message("user"):
        st.write(question)
    
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Query conversational RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Analyzing notes..."):
            try:
                response = conversational_rag_chain.invoke(
                    {"input": question},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )
                
                answer = response["answer"]
                st.write(answer)
                
                # Extract and clean sources
                sources = []
                if "context" in response and response["context"]:
                    for doc in response["context"]:
                        src_name = Path(doc.metadata.get("source", "Unknown")).name
                        page = doc.metadata.get("page", 0) + 1
                        source_str = f"{src_name} (Page {page})"
                        if source_str not in sources:
                            sources.append(source_str)
                
                if sources:
                    sources_html = "".join([f'<span class="source-badge">📖 {src}</span>' for src in sources])
                    st.markdown(f'<div style="margin-top: 10px;">{sources_html}</div>', unsafe_allow_html=True)
                
                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                err_msg = f"An error occurred: {e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
