# Hybrid Search RAG Chatbot

A Streamlit-based hybrid RAG chatbot that ingests PDFs or blog URLs, chunks them contextually, performs semantic search + BM25 retrieval, reranks results, and generates grounded answers through an OpenAI-compatible OpenRouter model using LangChain and LangGraph.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in `.env` with your OpenRouter credentials, then run:

```powershell
streamlit run app.py
```

If Streamlit prints large `transformers` / `torchvision` watcher errors on Windows, this project already disables the file watcher via `.streamlit/config.toml`. If you still see it in your shell, run:

```powershell
streamlit run app.py --server.fileWatcherType none
```

For API mode:

```powershell
uvicorn api:app --reload
```

## Features

- PDF and URL ingestion
- Contextual chunking with overlap
- Chroma vector search
- BM25 keyword search
- Merge, deduplication, and reranking
- LangGraph workflow with optional query rewriting
- Session-level conversational memory
- Persistent session storage in `storage/sessions.json`
- Request trace logging in `storage/traces`
- Source-aware answers in a polished Streamlit UI
- FastAPI endpoints for `/chat`, `/ingest`, and `/session/{session_id}`
