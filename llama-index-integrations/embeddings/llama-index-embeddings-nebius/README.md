# LlamaIndex Embeddings Integration: [Nebius AI Studio](https://studio.nebius.ai/)

## Overview

Integrate with Nebius AI Studio API, which provides access to open-source state-of-the-art text embeddings models.

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
