# About PaperBreeze⚡

## Overview

**PaperBreeze** is a Retrieval-Augmented Generation (RAG) system with an agentic architecture designed to simplify research papers for enthusiasts. Built using **Langgraph**, **FastAPI**, **Pinecone** vector database, and **Guardrails-AI**, for this works only on the text part of the PDF.

---

### Features:🌟
- **Efficient**: Processes documents of up to 10 pages.
- **Space-Efficient**: Deletes vectors and documents when the user ends the session with `END CHAT`.
- **Safe and Secure**: Utilizes **Guardrails-AI** to ensure all inputs and outputs are free from inappropriate, biased, or offensive content.

---

### Screenshots:
<image src='guardrails.png'>
<image src='index.png'>
  
---

### Usage:

Follow these steps to set up and run **PaperBreeze** locally:

### 1. Clone the Repository
```bash
git clone <repository_url>
cd PaperBreeze
```

### 2. Create virutal environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Create .env folder and paste your api keys
```bash
Get the following api keys
- GROQ
- PINECONE
- PINECONE_ENVIRONMENT
- HF_TOKEN
```

### 4. Navigate to app.py
```bash
Run uvicorn app:app --reload
```
