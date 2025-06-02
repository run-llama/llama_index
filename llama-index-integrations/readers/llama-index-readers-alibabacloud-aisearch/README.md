# LlamaIndex Readers Integration: Alibabacloud_Aisearch

## Installation

```
pip install llama-index-readers-alibabacloud-aisearch
```

## Usage

Supported file types: ppt/pptx, doc/docx, pdf, images and so on.
For further details, please visit:

- [document-analyze-api-details](https://help.aliyun.com/zh/open-search/search-platform/developer-reference/api-details)
- [image-analyze-api-details](https://help.aliyun.com/zh/open-search/search-platform/developer-reference/opensearch-api-details)

You can specify the `endpoint` and `aisearch_api_key` in the constructor, or set the environment variables `AISEARCH_ENDPOINT` and `AISEARCH_API_KEY`.

### Read local files

```python
from llama_index.readers.alibabacloud_aisearch import (
    AlibabaCloudAISearchDocumentReader,
    AlibabaCloudAISearchImageReader,
)
from llama_index.core import SimpleDirectoryReader

document_reader = AlibabaCloudAISearchDocumentReader()
image_reader = AlibabaCloudAISearchImageReader()

file_extractor = {}
for suffix in (".pdf", ".docx", ".doc", ".ppt", ".pptx"):
    file_extractor[suffix] = document_reader
for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
    file_extractor[suffix] = image_reader

documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data(show_progress=True)
print(documents)
```

### Read remote files

```python
from llama_index.readers.alibabacloud_aisearch import (
    AlibabaCloudAISearchImageReader,
)

image_reader = AlibabaCloudAISearchImageReader(
    service_id="ops-image-analyze-ocr-001"
)
image_urls = [
    "https://img.alicdn.com/imgextra/i1/O1CN01WksnF41hlhBFsXDNB_!!6000000004318-0-tps-1000-1400.jpg",
]

# The file_type is automatically determined based on the file extension.
# If it cannot be identified, manual specification of the file_type is required.
documents = image_reader.load_data(file_path=image_urls, file_type="jpg")
print(documents)
```
