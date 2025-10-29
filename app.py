import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from PIL import Image

from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, UnstructuredURLLoader, UnstructuredImageLoader
)
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=512
)
# Initialize embeddings and vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
DB_PATH = "vector_index"

# Utility: load documents from uploaded files or links
def load_documents(uploaded_files, urls):
    docs = []

    for file in uploaded_files:
        file_name = file.name
        with open(file_name, "wb") as f:
            f.write(file.read())

        if file_name.endswith(".txt"):
            docs.extend(TextLoader(file_name).load())
        elif file_name.endswith(".pdf"):
            docs.extend(PyPDFLoader(file_name).load())
        elif file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            docs.extend(UnstructuredImageLoader(file_name).load())

    if urls:
        docs.extend(UnstructuredURLLoader(urls=urls).load())

    return docs


# Update FAISS vector database
def update_vector_db(uploaded_files, urls):
    docs = load_documents(uploaded_files, urls)
    if not docs:
        st.warning("No documents found to process.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    if os.path.exists(DB_PATH):
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(split_docs)
    else:
        db = FAISS.from_documents(split_docs, embeddings)

    db.save_local(DB_PATH)
    st.success(f"✅ Added {len(split_docs)} document chunks to the knowledge base.")
    return db


# Retrieve relevant chunks and query Groq
def query_groq_llama(query, db):
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are an intelligent AI assistant powered by Groq LPU.
Use the following context from the user's documents to answer accurately and clearly.

Context:
{context}

User question: {query}
"""

    # Corrected invocation
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)



# ---------------- STREAMLIT UI -----------------
st.set_page_config(page_title="Groq RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖Project-6 Groq + LangChain RAG Chatbot - (Multiple files)")
st.markdown("Upload PDFs, text files, images, or URLs — then chat with your AI assistant powered by **Groq LLaMA**.")

# Sidebar upload section
st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload text, PDF, or image files",
    type=["txt", "pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)
urls_input = st.sidebar.text_area("Add URLs (comma separated):")

if st.sidebar.button("🔄 Update Knowledge Base"):
    urls = [u.strip() for u in urls_input.split(",") if u.strip()]
    update_vector_db(uploaded_files, urls)

# Load or initialize database
if os.path.exists(DB_PATH):
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    db = None

# Chat section
st.subheader("💬 Chat Interface")
user_query = st.text_input("Ask something about your uploaded documents:")

if st.button("Ask"):
    if not user_query:
        st.warning("Please enter a question.")
    elif db is None:
        st.warning("Knowledge base is empty. Please upload documents first.")
    else:
        with st.spinner("Thinking..."):
            answer = query_groq_llama(user_query, db)
            st.markdown(f"**🤖 Answer:** {answer}")

# Display image preview if uploaded
if uploaded_files:
    for file in uploaded_files:
        if file.name.lower().endswith((".png", ".jpg", ".jpeg")):
            st.image(Image.open(file), caption=f"Uploaded: {file.name}", use_container_width=True)
