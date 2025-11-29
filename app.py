import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import HumanMessage, SystemMessage

# Load API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit Page Setup
st.set_page_config(page_title="Knowledge Base AI Agent", layout="wide")
st.title("📚 Knowledge Base AI Agent")
st.write("Ask questions based on the uploaded company documents.")

# Load Vector Store
@st.cache_resource
def load_vectorstore(persist_directory="chroma_db"):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return store

# Generate response from OpenAI
def generate_answer(retrieved_docs, question):
    system_prompt = (
        "You are an intelligent assistant. Use ONLY the context provided. "
        "If the answer cannot be found in the context, reply: 'Information not available in documents.'\n"
    )
    context = "\n\n".join(
        f"[{i+1}] {doc.page_content[:700]}" for i, doc in enumerate(retrieved_docs)
    )
    prompt = f"{system_prompt}\nContext:\n{context}\n\nQuestion: {question}"

    model = ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    response = model([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ])

    return response.content

# UI
store = load_vectorstore()

question = st.text_input("🔍 Ask a question:")

top_k = st.slider("Number of document chunks to search", 1, 6, 3)

if st.button("Get Answer") and question:
    docs_and_scores = store.similarity_search_with_score(question, k=top_k)
