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
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel

_ = load_dotenv(find_dotenv())

model = DeepInfraEmbeddingModel()
response = model.get_query_embedding("hello world")
# Print the embeddings
print(response)
```

### Use with custom model_id

```python
from llama_index.embeddings.deepinfra import DeepinfraEmbeddings


model = DeepInfraEmbeddingModel(
    model_id="BAAI/bge-large-en-v1.5",
    api_token="YOUR_API_TOKEN",
    normalize=True,
)

response = model.get_query_embedding("hello world")
# Print the embeddings
print(response)
```

### Use with query prefix

```python
from dotenv import load_dotenv, find_dotenv
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel

_ = load_dotenv(find_dotenv())

model = DeepInfraEmbeddingModel(
    model_id="BAAI/bge-large-en-v1.5",
    query_prefix="query: ",
)

response = model.get_query_embedding("hello world")
# Print the embeddings
print(response)
```

### Use with text prefix

```python
from dotenv import load_dotenv, find_dotenv
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel

_ = load_dotenv(find_dotenv())

model = DeepInfraEmbeddingModel(
    model_id="BAAI/bge-large-en-v1.5",
    text_prefix="text: ",
)

response = model.get_text_embedding("hello world")
# Print the embeddings
print(response)
```

### Send batch requests

```python
from dotenv import load_dotenv, find_dotenv
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel

_ = load_dotenv(find_dotenv())

model = DeepInfraEmbeddingModel(
    model_id="BAAI/bge-large-en-v1.5",
)

texts = ["hello world", "goodbye world"]

response = model.get_text_embedding_batch(texts)
# Print the embeddings
print(response)
```

### Asynchronous requests

```python
from dotenv import load_dotenv, find_dotenv
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel

_ = load_dotenv(find_dotenv())

model = DeepInfraEmbeddingModel(
    model_id="BAAI/bge-large-en-v1.5",
)


async def main():
    text = "hello world"
    response = await model.aget_text_embedding(text)
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```
