# LlamaIndex Postprocessor Integration: DashScope-Rerank

The `llama-index-postprocessor-dashscope-rerank` package contains LlamaIndex integrations for the `gte-rerank` series models provided by Alibaba Tongyi Laboratory.

## Installation

```shell
pip install --upgrade llama-index llama-index-core llama-index-postprocessor-dashscope-rerank
```

## Setup

**Get started:**

1. Obtain the **API-KEY** from the [Alibaba Cloud ModelStudio platform](https://help.aliyun.com/document_detail/2712195.html?spm=a2c4g.2587460.0.i6).
2. Set **API-KEY**

```shell
export DASHSCOPE_API_KEY=YOUR_DASHSCOPE_API_KEY
```

**Example:**

```python
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank

nodes = [
    NodeWithScore(node=Node(text="text1"), score=0.7),
    NodeWithScore(node=Node(text="text2"), score=0.8),
]

dashscope_rerank = DashScopeRerank(top_n=5)
results = dashscope_rerank.postprocess_nodes(nodes, query_str="<user query>")
for res in results:
    print("Text: ", res.node.get_content(), "Score: ", res.score)
```

**output**

```text
Text:  text1 Score:  0.25589250620997755
Text:  text2 Score:  0.18071043165292258
```

### Parameters

|       Name       |  Type  |                                                                                                             Description                                                                                                              |   Default    |
| :--------------: | :----: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------: |
|      model       | `str`  |                                                                                                              model name                                                                                                              | `gte-rerank` |
|      top_n       | `int`  | The number of top documents to be returned in the ranking; if not specified, all candidate documents will be returned. If the specified top_n value exceeds the number of input candidate documents, all documents will be returned. |     `3`      |
| return_documents | `bool` |                                                    Whether to return the original text for each document in the returned sorted result list, with the default value being False.                                                     |   `False`    |
|     api_key      | `str`  |                                                                                                        The DashScope api key.                                                                                                        |    `None`    |
