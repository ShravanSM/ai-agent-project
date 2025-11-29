import os
import tempfile
import shutil
import streamlit as st

# LangChain v1 integrations
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

# Document loaders & utilities
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Basic page config
st.set_page_config(page_title="Knowledge Base AI Agent", layout="wide")
st.title("📚 Knowledge Base AI Agent")
st.write("Ask questions based on the uploaded company documents.")

# Read OPENAI key from env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY is not set. Set it in your shell before running the app.")

# Cache vector store loader
@st.cache_resource
def load_vectorstore(persist_directory="chroma_db"):
    # If DB exists, open it; otherwise return None
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        return None
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return store

# Utility: process uploaded files and persist to Chroma
def process_and_persist(uploaded_files, persist_directory="chroma_db"):
    all_docs = []

    for uploaded_file in uploaded_files:
        # Save to a temp file because loaders expect a filesystem path
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
            elif suffix == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
                docs = loader.load()
            elif suffix == ".docx":
                loader = Docx2txtLoader(tmp_path)
                docs = loader.load()
            else:
                st.error(f"Unsupported file type: {uploaded_file.name}")
                docs = []

            all_docs.extend(docs)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if not all_docs:
        st.error("No documents were loaded.")
        return None

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    # Create embeddings and persist to Chroma
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db

# UI: file uploader
uploaded_files = st.file_uploader("Upload documents (pdf, txt, docx)", type=["pdf", "txt", "docx"], accept_multiple_files=True)

if uploaded_files and st.button("Process & Index Documents"):
    with st.spinner("Processing and indexing documents — this may take a moment..."):
        db = process_and_persist(uploaded_files)
        if db:
            st.success("Documents processed and stored in Chroma (chroma_db). Refresh the page to load the DB.")
            st.stop()

# Load existing vectorstore (if any)
store = load_vectorstore()
if store is None:
    st.info("No indexed documents found. Upload documents to create the vector store.")

# Question UI
st.markdown("---")
question = st.text_input("🔎 Ask a question:")
col1, col2 = st.columns([1, 3])
with col1:
    top_k = st.slider("Number of document chunks to search", min_value=1, max_value=6, value=1)
with col2:
    st.write("")

if st.button("Get Answer"):
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is not set — set it in your shell and restart the app.")
    elif not question:
        st.error("Please enter a question first.")
    elif store is None:
        st.error("No indexed documents available. Upload and index documents first.")
    else:
        with st.spinner("Finding relevant passages and generating an answer..."):
            try:
                docs_and_scores = store.similarity_search_with_score(question, k=top_k)
                if not docs_and_scores:
                    st.info("No relevant documents found.")
                else:
                    retrieved_docs = [d for d, s in docs_and_scores]

                    # Build context from retrieved docs
                    context = "\n\n".join(f"[{i+1}] {doc.page_content[:1500]}" for i, doc in enumerate(retrieved_docs))

                    system_prompt = (
                        "You are an intelligent assistant. Use ONLY the context provided below to answer the question. "
                        "If the answer cannot be found in the context, reply: 'Information not available in documents.'\n"
                    )

                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
                    ]

                    chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
                    response = chat(messages)

                    st.subheader("Answer")
                    st.write(response.content)

                    st.subheader("Sources (retrieved chunks)")
                    for i, (doc, score) in enumerate(docs_and_scores):
                        st.markdown(f"**Chunk {i+1} — score:** {score:.4f}")
                        preview = doc.page_content[:1000].replace("\n", " ")
                        st.code(preview)

            except Exception as e:
                st.error(f"Error while generating answer: {e}")

# Small footer/help text
st.markdown("---")
st.caption("Upload documents, index them, then ask questions. The app persists the vector DB under 'chroma_db' in the project folder.")

