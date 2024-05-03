# LlamaIndex Embeddings Integration: Deepinfra

## Installation

```bash
pip install llama-index llama-index-embeddings-deepinfra
```

## Usage

### Use with default configuration

```python
from llama_index.embeddings.deepinfra import DeepinfraEmbeddings

model = DeepinfraEmbeddings()
response = model.get_query_embedding("hello world")
# Print the embeddings
print(response)
```

### Use with custom model_id

```python
from llama_index.embeddings.deepinfra import DeepinfraEmbeddings


model = DeepinfraEmbeddings(
    model_id="BAAI/bge-large-en-v1.5",
    api_token="YOUR_API_TOKEN",
    normalize=True,
)

response = model.get_query_embedding("hello world")
# Print the embeddings
print(response)
```
