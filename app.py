# app.py
import os
import streamlit as st

# NOTE: depending on your installed packages / versions, these imports may differ.
# If you get an ImportError, try:
#   from langchain.embeddings import OpenAIEmbeddings
#   from langchain.vectorstores import Chroma
#   from langchain.chat_models import ChatOpenAI
#
# I'm keeping the names you used originally; change them if necessary.
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

# Load API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Basic check
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY environment variable not found. Set it before running the app.")
    # allow the app to continue (useful for development), but some features will fail

# Streamlit Page Setup
st.set_page_config(page_title="Knowledge Base AI Agent", layout="wide")
st.title("📚 Knowledge Base AI Agent")
st.write("Ask questions based on the uploaded company documents. Answers will use ONLY the returned context.")

# Load Vector Store (cached)
@st.cache_resource
def load_vectorstore(persist_directory="chroma_db"):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return store

# Generate response from OpenAI using only provided context
def generate_answer(retrieved_docs, question):
    system_prompt = (
        "You are an intelligent assistant. Use ONLY the context provided. "
        "If the answer cannot be found in the context, reply: 'Information not available in documents.'\n"
    )

    # Build context string from the retrieved docs (truncate long chunks to keep prompt size reasonable)
    context = "\n\n".join(
        f"[{i+1}] {doc.page_content[:1500]}" for i, doc in enumerate(retrieved_docs)
    )

    prompt = f"{system_prompt}\nContext:\n{context}\n\nQuestion: {question}"

    model = ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    # Provide system and human messages (keeps instruction explicit)
    response = model([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ])

    # model(...) returns an object; we expect `.content` to hold the reply
    # If the return type differs in your version of langchain, you may need to access response[0].content or str(response)
    try:
        return response.content
    except Exception:
        # fallback: try indexing (some versions return list-like)
        try:
            return response[0].content
        except Exception:
            return str(response)

# UI: load store and interact
try:
    store = load_vectorstore()
except Exception as e:
    st.error(f"Failed to load vector store: {e}")
    st.stop()

# Input controls
question = st.text_input("🔎 Ask a question:")
top_k = st.slider("Number of document chunks to search (k)", 1, 10, 3)

if st.button("Get Answer") and question:
    with st.spinner("Searching documents and generating answer..."):
        try:
            docs_and_scores = store.similarity_search_with_score(question, k=top_k)
        except Exception as e:
            st.error(f"Error running similarity search: {e}")
            st.stop()

        if not docs_and_scores:
            st.info("No similar documents found in the vector store.")
        else:
            # docs_and_scores is list of (doc, score) tuples — extract docs and scores
            retrieved_docs = [d for d, s in docs_and_scores]
            scores = [s for d, s in docs_and_scores]

            # Generate answer using the retrieved docs
            answer = generate_answer(retrieved_docs, question)

            # Show answer
            st.subheader("Answer")
            st.write(answer)

            # Show source chunks & similarity scores
            with st.expander("Show source document chunks and similarity scores"):
                for i, (doc, score) in enumerate(docs_and_scores):
                    st.markdown(f"**Source [{i+1}] — score:** `{score}`")
                    # show a truncated preview
                    preview = doc.page_content if len(doc.page_content) < 1000 else doc.page_content[:1000] + "... (truncated)"
                    st.code(preview, language="text")
                    st.markdown("---")

            # Optional: allow user to copy context or full docs
            if st.checkbox("Show full context used for generation"):
                st.text_area("Full context", value="\n\n".join(d.page_content for d in retrieved_docs), height=300)

else:
    st.info("Enter a question and press **Get Answer** to query the knowledge base.")
