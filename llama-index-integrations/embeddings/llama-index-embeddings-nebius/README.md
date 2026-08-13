# LlamaIndex Embeddings Integration: [Nebius Token Factory](https://tokenfactory.nebius.com/)

## Overview

Integrate with the OpenAI-compatible Embeddings API in Nebius Token Factory, which provides access to open-source text embedding models.

## Installation

```bash
pip install llama-index-embeddings-nebius
```

## Usage

### Initialization

#### With environmental variables.

```.env
NEBIUS_API_KEY=your_api_key

```

```python
from llama_index.embeddings.nebius import NebiusEmbedding

embed_model = NebiusEmbedding(model_name="BAAI/bge-en-icl")
```

#### Without environmental variables

```python
from llama_index.embeddings.nebius import NebiusEmbedding

embed_model = NebiusEmbedding(
    api_key="your_api_key", model_name="BAAI/bge-en-icl"
)
```

The integration defaults to `https://api.tokenfactory.nebius.com/v1`. You can
set `api_base` explicitly to target a Dedicated Endpoint or another compatible
deployment. See the [Token Factory model list](https://docs.tokenfactory.nebius.com/models)
for currently available model IDs.

### Launching

#### Basic usage

```python
text = "Everyone loves justice at another person's expense"
embeddings = embed_model.get_text_embedding(text)
print(embeddings[:5])
```

#### Asynchronous usage

```python
text = "Everyone loves justice at another person's expense"
embeddings = await embed_model.aget_text_embedding(text)
print(embeddings[:5])
```

#### Batched usage

```python
texts = [
    "As the hours pass",
    "I will let you know",
    "That I need to ask",
    "Before I'm alone",
]

embeddings = embed_model.get_text_embedding_batch(texts)
print(*[x[:3] for x in embeddings], sep="\n")
```

#### Batched asynchronous usage

```python
texts = [
    "As the hours pass",
    "I will let you know",
    "That I need to ask",
    "Before I'm alone",
]

embeddings = await embed_model.aget_text_embedding_batch(texts)
print(*[x[:3] for x in embeddings], sep="\n")
```
