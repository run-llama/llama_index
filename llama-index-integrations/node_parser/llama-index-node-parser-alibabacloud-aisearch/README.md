# LlamaIndex Node_Parser Integration: Alibabacloud_Aisearch

## Installation

```
pip install llama-index-node-parser-alibabacloud-aisearch
```

## Optional Installation

For automatic parsing of image slices, you can optionally install `llama-index-readers-alibabacloud-aisearch`.

```
pip install llama-index-readers-alibabacloud-aisearch
```

## Usage

For further details, please visit [document-split-api-details](https://help.aliyun.com/zh/open-search/search-platform/developer-reference/document-split-api-details).

You can specify the `endpoint` and `aisearch_api_key` in the constructor, or set the environment variables `AISEARCH_ENDPOINT` and `AISEARCH_API_KEY`.

```python
from llama_index.node_parser.alibabacloud_aisearch import (
    AlibabaCloudAISearchNodeParser,
)
from llama_index.core import Document

try:
    from llama_index.readers.alibabacloud_aisearch import (
        AlibabaCloudAISearchImageReader,
    )

    image_reader = AlibabaCloudAISearchImageReader(
        service_id="ops-image-analyze-vlm-001"
    )
except ImportError:
    image_reader = None
node_parser = AlibabaCloudAISearchNodeParser(
    chunk_size=1024, image_reader=image_reader
)
nodes = node_parser(
    [
        Document(text="content1", mimetype="text/markdown"),
        Document(
            text="content2 ![IMAGE](https://img.alicdn.com/imgextra/i1/O1CN01WksnF41hlhBFsXDNB_!!6000000004318-0-tps-1000-1400.jpg)",
            mimetype="text/markdown",
        ),
    ],
    show_progress=True,
)
for i, node in enumerate(nodes):
    print(f"[SPLIT#{i}]:\n{node.get_content()}")
    print("-" * 80)
```
