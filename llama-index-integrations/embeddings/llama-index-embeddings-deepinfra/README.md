# LlamaIndex Embeddings Integration: Deepinfra

With this integration, you can use the Deepinfra embeddings model to get embeddings for your text data.
Here is the link to the [embeddings models](https://deepinfra.com./models/embeddings).

First, you need to sign up on the [Deepinfra website](https://deepinfra.com/) and get the API token.
You can copy model_ids over the model cards and start using them in your code.

## Installation

```bash
pip install llama-index llama-index-embeddings-deepinfra
```

## Usage

In order to provide the API token, you can set the `DEEPINFRA_API_TOKEN` environment variable or pass it as an argument to the `DeepinfraEmbeddings` class.

### Use with default configuration

```python
from dotenv import load_dotenv, find_dotenv
from llama_index.embeddings.deepinfra import DeepinfraEmbeddings

_ = load_dotenv(find_dotenv())

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
