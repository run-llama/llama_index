---
title: Dell PowerScale Readers
---

[Dell PowerScale](https://www.dell.com/en-us/shop/powerscale-family/sf/powerscale) is an enterprise scale-out storage system hosting the industry-leading OneFS filesystem, available on-prem or in the cloud.

These readers leverage unique PowerScale capabilities to determine which files have been modified since the last application run, returning only those changed files for processing. This eliminates the need to re-process (chunk and embed) unchanged files, improving overall data ingestion efficiency.

This integration requires PowerScale's MetadataIQ feature to be enabled. Additional information can be found on the GitHub repository: [https://github.com/dell/powerscale-rag-connector](https://github.com/dell/powerscale-rag-connector)

### Installation

```bash
pip install powerscale-rag-connector
```

## Usage

### PowerScaleSimpleDirectoryReader

`PowerScaleSimpleDirectoryReader` extends LlamaIndex's `SimpleDirectoryReader`, using MetadataIQ to identify changed files before reading and parsing their contents. Use `input_dir` to scan a directory or `input_files` for a specific list, the two are mutually exclusive.

```python
from powerscale_rag_connector import PowerScaleSimpleDirectoryReader

reader = PowerScaleSimpleDirectoryReader(
    es_host_url="http://elasticsearch:9200",  # Elasticsearch URI including port
    es_index_name="metadataiq",               # Elasticsearch index name
    es_api_key="your-api-key",                # encoded Elasticsearch API key
    input_dir="/ifs/data",                    # or use input_files=[...] for specific files
    force_scan=False,                         # set True for a full scan
)

for doc in reader.lazy_load_data():
    print(doc)
```

### PowerScaleUnstructuredReader

`PowerScaleUnstructuredReader` uses MetadataIQ to identify changed files and parses their contents using LlamaIndex's `UnstructuredReader`. Use `mode="elements"` for element-level documents, or `mode="single"` (default) for one document per file.

```python
from powerscale_rag_connector import PowerScaleUnstructuredReader

reader = PowerScaleUnstructuredReader(
    es_host_url="http://elasticsearch:9200",
    es_index_name="metadataiq",
    es_api_key="your-api-key",
    folder_path="/ifs/data",   # or dataset_name="my-dataset"
    mode="elements",           # use "single" for the entire file as one document
)

for doc in reader.lazy_load_data():
    print(doc)
```

Each returned [`Document`](https://docs.llamaindex.ai/en/stable/api_reference/schema/) includes `source`, `snapshot`, `lin` (OneFS logical inode number), and `change_types` (`ENTRY_ADDED` or `ENTRY_MODIFIED`) in its metadata. Your RAG application can use `change_types` to add or update entries in your vector store.

## Examples

- [PowerScale LlamaIndex SimpleDirectoryReader](https://github.com/dell/powerscale-rag-connector/blob/main/examples/powerscale_llamaindex_simple_directory_reader.py)
- [PowerScale LlamaIndex UnstructuredReader](https://github.com/dell/powerscale-rag-connector/blob/main/examples/powerscale_llamaindex_unstructured_reader.py)