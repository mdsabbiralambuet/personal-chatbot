# Sabbir's Personal Chatbot

A RAG-powered chatbot that answers questions about me, my research, and my projects.
Runs locally with Ollama; FastAPI backend, minimal HTML frontend.

## Stack
- **LLM:** Llama 3.2 (3B) via Ollama
- **Embeddings:** nomic-embed-text via Ollama
- **Vector store:** ChromaDB
- **Backend:** FastAPI
- **Frontend:** plain HTML/JS

## Setup

1. Install [Ollama](https://ollama.com) and pull the models:
```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
```

2. Install Python dependencies:
```bash
   pip install -r requirements.txt
```

3. Add your content to `data/` as markdown files, then build the index:
```bash
   python index.py
```

4. Run the server:
```bash
   uvicorn server:app --reload
```

5. Open http://127.0.0.1:8000 in your browser.