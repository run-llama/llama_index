# LlamaIndex Postprocessor Integration: Xinference Rerank

Xorbits Inference (Xinference) is an open-source platform to streamline the operation and integration of a wide array of AI models.

You can find a list of built-in rerank models in Xinference from its document [Rerank Models](https://inference.readthedocs.io/en/latest/models/builtin/rerank/index.html)

To learn more about Xinference in general, visit https://inference.readthedocs.io/en/stable/models/model_abilities/rerank.html

## Installation

```shell
pip install llama-index-postprocessor-xinference-rerank
```

## Usage

**Parameters Description:**

- `model`: Model uid not model name, sometimes they may be the same (e.g., `bge-reranker-base`).
- `base_url`: base url of Xinference (e.g., `http://localhost:9997`).
- `top_n`: Top n nodes to return from reranker. (default 5).

**Nodes Rerank Example**

```python
from llama_index.postprocessor.xinference_rerank import XinferenceRerank

xi_model_uid = "xinference model uid"
xi_base_url = "xinference base url"

xi_rerank = XinferenceRerank(
    top_n=5,
    model=xi_model_uid,
    base_url=xi_base_url,
)


def test_rerank_nodes(nodes, query_str):
    response = xi_rerank.postprocess_nodes(nodes, query_str)
```
