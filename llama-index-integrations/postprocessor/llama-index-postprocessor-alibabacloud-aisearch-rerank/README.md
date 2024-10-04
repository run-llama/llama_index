# LlamaIndex Postprocessor Integration: Alibabacloud_Aisearch_Rerank

## Installation

```bash
pip install llama-index-postprocessor-alibabacloud-aisearch-rerank
```

## Usage

For further details, please visit [ranker-api-details](https://help.aliyun.com/zh/open-search/search-platform/developer-reference/ranker-api-details).

You can specify the `endpoint` and `aisearch_api_key` in the constructor, or set the environment variables `AISEARCH_ENDPOINT` and `AISEARCH_API_KEY`.

```python
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.alibabacloud_aisearch_rerank import (
    AlibabaCloudAISearchRerank,
)

nodes = [
    NodeWithScore(
        node=Node(id_="1", text="<text1>"),
        score=0.7,
    ),
    NodeWithScore(
        node=Node(id_="2", text="<text2>"),
        score=0.8,
    ),
    NodeWithScore(
        node=Node(id_="3", text="<text3>"),
        score=0.1,
    ),
]
reranker = AlibabaCloudAISearchRerank(top_n=2)
new_nodes = reranker.postprocess_nodes(nodes, query_str="<query>")
for node in new_nodes:
    print(f"{node.node.text[:20]}\t{node.score}")
```
