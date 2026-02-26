# LlamaIndex Embeddings Integration: HuggingFace-LangChain

This package provides LlamaIndex embedding integration using [langchain-huggingface](https://python.langchain.com/docs/integrations/platforms/huggingface/), supporting both local sentence-transformers and remote HuggingFace Inference API embeddings.

## Installation

```bash
pip install llama-index-embeddings-huggingface-langchain
```

## Usage

### Local embeddings (sentence-transformers)

```python
from llama_index.embeddings.huggingface_langchain import HuggingFaceLangChainEmbedding

embed_model = HuggingFaceLangChainEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",
)

# Single text embedding
embedding = embed_model.get_text_embedding("Hello world")
print(f"Embedding dimensions: {len(embedding)}")

# Query embedding (optimized for retrieval)
query_embedding = embed_model.get_query_embedding("What is Python?")

# Batch embeddings
texts = ["First document", "Second document", "Third document"]
embeddings = embed_model.get_text_embedding_batch(texts)
```

### Remote embeddings (HuggingFace Inference API)

```python
embed_model = HuggingFaceLangChainEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    backend="api",
)

embedding = embed_model.get_text_embedding("Hello world")
```

### Parameters

- `model_name` (str): HuggingFace model name (default: `"sentence-transformers/all-mpnet-base-v2"`)
- `backend` (str): `"local"` for sentence-transformers, `"api"` for HuggingFace Inference API
- `huggingfacehub_api_token` (str): HuggingFace API token (falls back to `HF_TOKEN` env var)
- `encode_kwargs` (dict): Additional encoding arguments (e.g., `{"normalize_embeddings": True}`)
- `model_init_kwargs` (dict): Additional model initialization arguments

### Use with LlamaIndex

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

embed_model = HuggingFaceLangChainEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",
)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
```
