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
| **Output in this Reader** | Markdown only     | Markdown only                              |

> Note: MinerU Python SDK precision mode supports extra output formats
> (images/JSON/docx/html/latex), but `MinerUReader` in this integration
> currently returns only `result.markdown` as `Document.text`.

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

### Mixed Sources (local path + URL)

You can parse local files and remote URLs in one call:

```python
from llama_index.readers.mineru import MinerUReader

reader = MinerUReader()
documents = reader.load_data(
    [
        "/path/to/local_a.pdf",
        "/path/to/local_b.docx",
        "https://cdn-mineru.openxlab.org.cn/demo/example.pdf",
    ]
)

for doc in documents:
    print(doc.metadata["source"], "-", doc.text[:100])
```

### Attach custom metadata with `extra_info`

Use `extra_info` when you want to merge custom metadata fields into every
returned `Document.metadata` (for example project/tenant/tag). It does not
change parsing behavior or output format.

```python
from llama_index.readers.mineru import MinerUReader

reader = MinerUReader()
documents = reader.load_data(
    "/path/to/paper.pdf",
    extra_info={
        "project": "paper-rag",
        "tenant": "team-a",
        "source_type": "research_pdf",
    },
)

print(documents[0].metadata["project"])      # paper-rag
print(documents[0].metadata["source_type"])  # research_pdf
print(documents[0].text[:120])               # still Markdown text
```

### Per-Page Splitting

When `split_pages=True`, each PDF page becomes a separate Document — ideal for RAG pipelines that need page-level granularity.

```python
reader = MinerUReader(split_pages=True, pages="1-5")
documents = reader.load_data("/path/to/paper.pdf")

for doc in documents:
    print(f"Page {doc.metadata['page']}: {doc.text[:100]}...")
```

### Page Range + Split Pages (PDF only)

Use `pages` together with `split_pages=True` to parse only selected PDF pages.
In this mode, each selected page becomes one `Document`.

```python
from llama_index.readers.mineru import MinerUReader

reader = MinerUReader(
    mode="precision",
    token="your-api-token",  # or set MINERU_TOKEN
    pages="2-4",
    split_pages=True,
    language="en",
)

documents = reader.load_data("/path/to/paper.pdf")
for doc in documents:
    print(doc.metadata.get("page"), doc.metadata.get("source"))
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

### Reader initialization (`MinerUReader(...)`)

| Parameter     | Type          | Default   | Description                                                             |
| ------------- | ------------- | --------- | ----------------------------------------------------------------------- |
| `mode`        | `str`         | `"flash"` | Parsing mode: `"flash"` or `"precision"`                                |
| `token`       | `str \| None` | `None`    | MinerU API token (precision mode). Falls back to `MINERU_TOKEN` env var. Apply here: [https://mineru.net/apiManage/token](https://mineru.net/apiManage/token) |
| `language`    | `str`         | `"ch"`    | Document language code                                                  |
| `pages`       | `str \| None` | `None`    | Page range, e.g. `"1-10"`                                               |
| `timeout`     | `int`         | `600`     | Max seconds to wait for task completion                                 |
| `split_pages` | `bool`        | `False`   | Split PDF into per-page Documents                                       |
| `ocr`         | `bool`        | `False`   | Enable OCR (precision mode only)                                        |
| `formula`     | `bool`        | `True`    | Enable formula recognition (precision mode only)                        |
| `table`       | `bool`        | `True`    | Enable table recognition (precision mode only)                          |

### `load_data(...)` arguments

| Parameter    | Type                                 | Default | Description                                                                    |
| ------------ | ------------------------------------ | ------- | ------------------------------------------------------------------------------ |
| `sources`    | `str \| Path \| list[str \| Path]`   | —       | Single file path/URL, or a list of file paths/URLs                            |
| `extra_info` | `dict \| None`                       | `None`  | Custom metadata merged into each returned `Document.metadata`                  |

## Links

- [MinerU Official Website](https://mineru.net)
- [MinerU API Documentation](https://mineru.net/apiManage/docs)
- [MinerU Token Application](https://mineru.net/apiManage/token)
- [MinerU Python SDK](https://github.com/opendatalab/MinerU-Ecosystem/tree/main/sdk/python)
