# LlamaIndex Readers Integration: Plasmate

[Plasmate](https://plasmate.app) is an open-source browser engine that compiles HTML into a [Semantic Object Model (SOM)](https://www.w3.org/community/nicg/), producing **10-16x fewer tokens** than raw HTML while preserving document structure.

This reader fetches web pages using Plasmate and converts them into LlamaIndex `Document` objects — ideal for RAG pipelines that need clean, structured web content.

## Installation

```bash
pip install llama-index-readers-plasmate plasmate
```

## Usage

```python
from llama_index.readers.plasmate import PlasmateWebReader

reader = PlasmateWebReader()
documents = reader.load_data(urls=["https://example.com"])
index = VectorStoreIndex.from_documents(documents)
```

## Why Plasmate?

| Reader | Output | Token Efficiency |
|--------|--------|-----------------|
| `SimpleWebPageReader` | Raw HTML | 1x (baseline) |
| `BeautifulSoupWebReader` | Extracted text | ~3-5x |
| `TrafilaturaWebReader` | Main content text | ~5-8x |
| **`PlasmateWebReader`** | **Structured semantic content** | **10-16x** |

Plasmate doesn't just strip tags — it compiles HTML into a semantic representation that preserves headings, hierarchy, and document structure. This means better retrieval quality with dramatically fewer tokens.

## Configuration

```python
reader = PlasmateWebReader(
    timeout=30,       # seconds per page fetch
    javascript=True,  # enable JS rendering
)
```

## Links

- [Plasmate](https://plasmate.app) — Open-source browser engine
- [W3C New Interfaces Community Group](https://www.w3.org/community/nicg/) — Standards work
- [PyPI](https://pypi.org/project/plasmate/) | [npm](https://www.npmjs.com/package/plasmate) | [crates.io](https://crates.io/crates/plasmate)
- Apache 2.0 licensed
