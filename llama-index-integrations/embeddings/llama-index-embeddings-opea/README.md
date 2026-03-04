# LlamaIndex Embeddings Integration: OPEA Embeddings

OPEA (Open Platform for Enterprise AI) is a platform for building, deploying, and scaling AI applications. As part of this platform, many core gen-ai components are available for deployment as microservices, including LLMs.

Visit [https://opea.dev](https://opea.dev) for more information, and their [GitHub](https://github.com/opea-project/GenAIComps) for the source code of the OPEA components.

## Installation

1. Install the required Python packages:

```bash
%pip install llama-index-embeddings-opea
```

## Usage

```python
from llama_index.embeddings.opea import OPEAEmbedding

embed_model = OPEAEmbedding(
    model="<model_name>",
    api_base="http://localhost:8080/v1",
    embed_batch_size=10,
)

embeddings = embed_model.get_text_embedding("text")

embeddings = embed_model.get_text_embedding_batch(["text1", "text2"])
```
