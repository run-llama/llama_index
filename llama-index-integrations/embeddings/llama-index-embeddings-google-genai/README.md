# Google GenAI Embeddings

This package provides a wrapper around the Google GenAI API, allowing you to use Gemini and Vertex AI embeddings in your projects.

## Installation

```bash
pip install llama-index-embeddings-google-genai
```

## Usage

```python
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")

embeddings = embed_model.get_text_embedding("Hello, world!")
print(embeddings)
```

## Vertex AI

```python
embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    vertexai_config={
        "project": "your-project-id",
        "location": "your-location",
    },
)
```
