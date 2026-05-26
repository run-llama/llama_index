# LlamaIndex Embeddings Integration: Telnyx

[Telnyx](https://telnyx.com) provides an OpenAI-compatible Embeddings API with models like gte-large and e5-large.

This integration lets you use Telnyx-hosted embedding models with LlamaIndex for RAG, similarity search, and indexing workflows.

## Installation

```shell
pip install llama-index-embeddings-telnyx
```

## Setup

Get an API key from the [Telnyx Mission Control Portal](https://portal.telnyx.com/) and set it as an environment variable:

```bash
export TELNYX_API_KEY="KEY_ID_SECRET"
```

## Usage

### Basic Embedding

```python
from llama_index.embeddings.telnyx import TelnyxEmbedding

embed_model = TelnyxEmbedding(model_name="thenlper/gte-large")

embedding = embed_model.get_text_embedding("What is Telnyx?")
print(f"Embedding dimension: {len(embedding)}")
```

### Batch Embedding

```python
from llama_index.embeddings.telnyx import TelnyxEmbedding

embed_model = TelnyxEmbedding(model_name="thenlper/gte-large")

texts = ["First document.", "Second document.", "Third document."]
embeddings = embed_model.get_text_embedding_batch(texts)
print(f"Got {len(embeddings)} embeddings")
```

### With LlamaIndex Index

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.telnyx import TelnyxEmbedding

embed_model = TelnyxEmbedding(model_name="thenlper/gte-large")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

query_engine = index.as_query_engine()
response = query_engine.query("What is in these documents?")
print(response)
```

## Available Models

- `thenlper/gte-large`

See the full list at [developers.telnyx.com/docs/inference/models](https://developers.telnyx.com/docs/inference/models).

## Resources

- [Telnyx AI Inference Docs](https://developers.telnyx.com/docs/inference/getting-started)
- [Telnyx Model Catalog](https://developers.telnyx.com/docs/inference/models)
- [Telnyx Portal](https://portal.telnyx.com/)
