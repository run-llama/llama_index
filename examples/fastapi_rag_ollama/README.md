# FastAPI + LlamaIndex RAG Example (Ollama)

This example demonstrates how to build a simple
Retrieval-Augmented Generation (RAG) API using
LlamaIndex and FastAPI, powered by a **local LLM via Ollama**.

## Features
- Local, free LLM (no API keys required)
- Document ingestion and indexing
- Simple query API endpoint
- Clean, production-style structure

## Prerequisites
- Python 3.9+
- Ollama installed locally

## Setup

Pull a model using Ollama:
```bash
ollama pull llama3
````

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --reload
```
Alternatively, you can test the API using FastAPI's built-in Swagger UI:

Open your browser and visit:
http://localhost:8000/docs

## Example Request

```bash
curl -X POST "http://localhost:8000/query" \
-H "Content-Type: application/json" \
-d '{"query": "What is this example about?"}'
```

## Notes

* This example uses a local LLM by default.
* It is intended as a minimal, beginner-friendly
  demonstration of integrating LlamaIndex with FastAPI.