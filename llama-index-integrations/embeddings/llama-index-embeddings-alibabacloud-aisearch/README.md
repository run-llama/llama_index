# LlamaIndex Embeddings Integration: Alibabacloud_Aisearch

## Installation

```
pip install llama-index-embeddings-alibabacloud-aisearch
```

## Usage

For further details, please visit [text-embedding-api-details](`https://help.aliyun.com/zh/open-search/search-platform/developer-reference/text-embedding-api-details`).

You can specify the `endpoint` and `aisearch_api_key` in the constructor, or set the environment variables `AISEARCH_ENDPOINT` and `AISEARCH_API_KEY`.

```python
from llama_index.embeddings.alibabacloud_aisearch import (
    AlibabaCloudAISearchEmbedding,
)

embed_model = AlibabaCloudAISearchEmbedding()
embedding = embed_model.get_query_embedding("llama-index")
print(len(embedding))

# embeddings = embed_model.get_text_embedding_batch(
#     ["科学技术是第一生产力", "opensearch产品文档"], show_progress=True
# )
```
