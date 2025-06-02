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

```python
from dotenv import load_dotenv, find_dotenv
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel

# Load environment variables
_ = load_dotenv(find_dotenv())

# Initialize model with optional configuration
model = DeepInfraEmbeddingModel(
    model_id="BAAI/bge-large-en-v1.5",  # Use custom model ID
    api_token="YOUR_API_TOKEN",  # Optionally provide token here
    normalize=True,  # Optional normalization
    text_prefix="text: ",  # Optional text prefix
    query_prefix="query: ",  # Optional query prefix
)

# Example usage
response = model.get_text_embedding("hello world")

# Batch requests
texts = ["hello world", "goodbye world"]
response = model.get_text_embedding_batch(texts)

# Query requests
response = model.get_query_embedding("hello world")


# Asynchronous requests
async def main():
    text = "hello world"
    response = await model.aget_text_embedding(text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```
