# LlamaIndex Embeddings Integration: Xinference

Xorbits Inference (Xinference) is an open-source platform to streamline the operation and integration of a wide array of AI models.

You can find a list of built-in embedding models in Xinference from its document [Embedding Models](https://inference.readthedocs.io/en/latest/models/builtin/embedding/index.html)

To learn more about Xinference in general, visit https://inference.readthedocs.io/en/latest/

## Installation

```shell
pip install llama-index-embeddings-xinference
```

## Usage

**Parameters Description:**

- `model_uid`: Model uid not the model name, sometimes they may be the same (e.g., `bce-embedding-base_v1`).
- `base_url`: base url of Xinference (e.g., `http://localhost:9997`).
- `timeout`: request timeout set (default 60s).
- `prompt`: Text to embed.

**Text Embedding Example**

```python
from llama_index.embeddings.xinference import XinferenceEmbedding

xi_model_uid = "xinference model uid"
xi_base_url = "xinference base url"

xi_embed = XinferenceEmbedding(
    model_uid=xi_model_uid,
    base_url=xi_base_url,
    timeout=60,
)


def text_embedding(prompt: str):
    embeddings = xi_embed.get_query_embedding(prompt)
    print(embeddings)


async def async_text_embedding(prompt: str):
    embeddings = await xi_embed.aget_query_embedding(prompt)
    print(embeddings)
```
