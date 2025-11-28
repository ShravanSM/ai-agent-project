import os
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_documents(folder="sample_docs"):
    docs = []
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder} (Place your PDFs/TXT files here)")
        return docs

    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif fname.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf8")
            docs.extend(loader.load())
    return docs

def create_vectorstore(docs, persist_directory="chroma_db"):
    if len(docs) == 0:
        print("No documents found to index.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    store.persist()
    print(f"Vector store saved to: {persist_directory}")
    return store

if __name__ == "__main__":
    documents = load_documents()
    create_vectorstore(documents)
    print("Indexing complete.")
