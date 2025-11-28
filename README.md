# KnowledgeBase Agent — AI Agent Challenge

## Overview
This project is a simple **KnowledgeBase AI Agent** built for the 48-Hour AI Agent Development Challenge.

It allows users to:
- Upload or store company documents (PDF/TXT)
- Convert them into vector embeddings
- Perform semantic search
- Ask natural-language questions
- Get answers with source references

This agent is useful for:
✔ HR document lookup  
✔ Company policy FAQs  
✔ Support knowledge base  
✔ Operations document search  

 ## Repository Structure

ai-agent-project/
├─ app.py
├─ indexer.py
├─ requirements.txt
├─ README.md
├─ sample_docs/
│   └─ sample1.txt
├─ .gitignore
└─ chroma_db/
## Features

- Ingests TXT and PDF documents from the `sample_docs/` folder
- Splits documents into chunks and creates vector embeddings
- Stores embeddings in a local Chroma vector database
- Streamlit UI to ask natural-language questions
- Retrieves relevant chunks using semantic search
- Generates accurate answers with source references

## Quick Start (How to Run This Project)

Follow these simple steps to run the KnowledgeBase Agent on your computer.

### 1. Create a virtual environment
Windows:
python -m venv .venv
.venv\Scripts\activate

Mac/Linux:
python3 -m venv .venv
source .venv/bin/activate

### 2. Install the required packages
pip install -r requirements.txt

### 3. Set your OpenAI API Key
Windows (PowerShell):
$env:OPENAI_API_KEY="your_api_key"

Mac/Linux:
export OPENAI_API_KEY="your_api_key"

### 4. Create the vector database
python indexer.py

### 5. Run the Streamlit app
streamlit run app.py

Now open the link shown in the terminal (usually http://localhost:8501) and ask questions.

## Deployment (How to Publish This App Online)

To make this project available online, follow these easy steps:

### 1. Go to Streamlit Cloud
Visit: https://share.streamlit.io  
Sign in with your GitHub account.

### 2. Create a new app
- Click **“New app”**
- Select this repository: `ai-agent-project`
- Choose branch: `main`
- Choose file to run: `app.py`

### 3. Add your OpenAI API Key (very important)
In Streamlit Cloud:
- Open **App Settings**
- Go to **Secrets**
- Add this:
OPENAI_API_KEY = "your_api_key_here"

### 4. Deploy the app
Click **Deploy**.  
Streamlit Cloud will install the dependencies and start your app.

### 5. Copy your public demo link
Once deployed, Streamlit will provide a public URL.  
This is the link you will submit in the challenge form.
## Limitations & Future Improvements

### Current Limitations
- The Chroma vector database is stored locally.
- The app cannot upload files through the UI. Indexing must be done through `indexer.py`.
- Requires an OpenAI API key.
- No login or authentication system.

### Future Improvements
- Add file upload support inside Streamlit.
- Move vector store to cloud DB (Pinecone / Supabase / Weaviate).
- Add chat history & conversation memory.
- Add user authentication (login system).
- Improve UI design and theme.
## Credits
This project was created by **ShravanSM**  
as part of the **48-Hour AI Agent Development Challenge**.



