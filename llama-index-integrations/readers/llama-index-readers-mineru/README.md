# MinerU Reader

```bash
pip install llama-index-readers-mineru
```

This reader uses the [MinerU](https://mineru.net) document parsing API to extract high-quality Markdown from PDF, Doc/Docx, PPT/PPTx, images, and Excel files. It supports two parsing modes:

| Feature                   | Flash (default)   | Precision                                  |
| ------------------------- | ----------------- | ------------------------------------------ |
| **Auth**                  | No token required | Token required                             |
| **Speed**                 | Blazing fast      | Standard                                   |
| **File size**             | Max 10 MB         | Max 200 MB                                 |
| **Page limit**            | Max 20 pages      | Max 600 pages                              |
| **OCR / Formula / Table** | Disabled          | Configurable                               |
| **Output**                | Markdown only     | Markdown + images, JSON, docx, html, latex |

## Usage

### Flash Mode (default, no token needed)

```python
from llama_index.readers.mineru import MinerUReader

reader = MinerUReader()

# Parse a single PDF from URL
documents = reader.load_data(
    "https://cdn-mineru.openxlab.org.cn/demo/example.pdf"
)
print(documents[0].text)

# Parse a local file
documents = reader.load_data("/path/to/local.pdf")

# Parse multiple files at once
documents = reader.load_data(
    [
        "https://cdn-mineru.openxlab.org.cn/demo/example.pdf",
        "/path/to/local.pdf",
    ]
)
```

### Precision Mode (token required)

Get your free token from [MinerU API Management](https://mineru.net/apiManage/token).

```python
from llama_index.readers.mineru import MinerUReader

reader = MinerUReader(
    mode="precision",
    token="your-api-token",  # or set MINERU_TOKEN env var
    ocr=True,
    formula=True,
    table=True,
    language="en",
    pages="1-20",
)

documents = reader.load_data("/path/to/scanned_paper.pdf")
```

### Per-Page Splitting

When `split_pages=True`, each PDF page becomes a separate Document — ideal for RAG pipelines that need page-level granularity.

```python
reader = MinerUReader(split_pages=True, pages="1-5")
documents = reader.load_data("/path/to/paper.pdf")

for doc in documents:
    print(f"Page {doc.metadata['page']}: {doc.text[:100]}...")
```

### Use with LlamaIndex Pipeline

```python
from llama_index.core import VectorStoreIndex
from llama_index.readers.mineru import MinerUReader

reader = MinerUReader()
documents = reader.load_data("/path/to/paper.pdf")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("Summarize the key findings")
print(response)
```

## Parameters

| Parameter     | Type          | Default   | Description                                                             |
| ------------- | ------------- | --------- | ----------------------------------------------------------------------- |
| `mode`        | `str`         | `"flash"` | Parsing mode: `"flash"` or `"precision"`                                |
| `token`       | `str \| None` | `None`    | MinerU API token (precision mode). Falls back to `MINERU_TOKEN` env var |
| `language`    | `str`         | `"ch"`    | Document language code                                                  |
| `pages`       | `str \| None` | `None`    | Page range, e.g. `"1-10"`                                               |
| `timeout`     | `int`         | `600`     | Max seconds to wait for task completion                                 |
| `split_pages` | `bool`        | `False`   | Split PDF into per-page Documents                                       |
| `ocr`         | `bool`        | `False`   | Enable OCR (precision mode only)                                        |
| `formula`     | `bool`        | `True`    | Enable formula recognition (precision mode only)                        |
| `table`       | `bool`        | `True`    | Enable table recognition (precision mode only)                          |

## Links

- [MinerU Official Website](https://mineru.net)
- [MinerU API Documentation](https://mineru.net/apiManage/docs)
- [MinerU Python SDK](https://github.com/opendatalab/MinerU-Ecosystem/tree/main/sdk/python)
