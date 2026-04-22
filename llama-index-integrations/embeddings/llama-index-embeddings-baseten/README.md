# LlamaIndex Embeddings Integration: Baseten

This integration allows you to use Baseten's hosted models with LlamaIndex.

## Installation

Install the required packages:

```bash
pip install llama-index-embeddings-baseten
pip install llama-index
```

Baseten embeddings are offered through dedicated deployments. You need to deploy your preferred embeddings model in your Baseten dashboard and provide the 8 character model id like `abcd1234`.

## Usage

```python
from llama_index.embeddings.baseten import BasetenEmbedding

# Using dedicated endpoint
# You can find the model_id by in the Baseten dashboard here: https://app.baseten.co/overview
embed_model = BasetenEmbedding(
    model_id="MODEL_ID",
    api_key="YOUR_API_KEY",
)

# Single embedding
embedding = embed_model.get_text_embedding("Hello, world!")

# Batch embeddings
embeddings = embed_model.get_text_embedding_batch(
    ["Hello, world!", "Goodbye, world!"]
)
```
