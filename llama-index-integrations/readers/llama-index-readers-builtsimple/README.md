# llama-index-readers-builtsimple

LlamaIndex readers for [Built-Simple](https://built-simple.ai) research APIs, providing semantic search over scientific literature.

[![PyPI version](https://badge.fury.io/py/llama-index-readers-builtsimple.svg)](https://pypi.org/project/llama-index-readers-builtsimple/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **PubMed Reader** - 4.5M+ biomedical articles with hybrid semantic/keyword search
- **ArXiv Reader** - 2.7M+ preprints in physics, math, CS, and ML
- **Wikipedia Reader** - Semantic search over Wikipedia articles
- **No API key required** - Free tier available for all endpoints
- **Rich metadata** - Full citation info for all documents

## What Data is Included

### PubMed Reader

Each document contains:

- **Text**: Title + abstract (default) OR **full article text** (with `include_full_text=True`)
- **Metadata**:
  - `pmid` - PubMed ID (e.g., "31041627")
  - `title` - Full article title
  - `journal` - Publication journal name
  - `year` - Publication year
  - `doi` - DOI identifier
  - `doi_url` - Direct DOI link
  - `url` - Link to PubMed page
  - `has_full_text` - Boolean indicating if full text was fetched
  - `full_text_length` - Character count of full text (when available)

**🔥 FULL TEXT AVAILABLE!** Unlike most research APIs that only provide abstracts, Built-Simple has full article text for millions of papers:

```python
# Get full article text (15K-70K chars per article)
reader = BuiltSimplePubMedReader(include_full_text=True)
docs = reader.load_data("cancer immunotherapy", limit=5)

for doc in docs:
    print(f"Full text length: {len(doc.text)} chars")  # ~15,000-70,000 chars!
```

### ArXiv Reader

Each document contains:

- **Text**: Title + authors + full abstract
- **Metadata**:
  - `arxiv_id` - ArXiv identifier (e.g., "2301.12345" or "cs/0308031")
  - `title` - Paper title
  - `authors` - Author names
  - `year` - Publication year
  - `url` - Link to ArXiv abstract page
  - `pdf_url` - Direct PDF download link
  - `similarity_score` - Semantic relevance score (0-1)

**Note**: Full paper PDFs are NOT downloaded—only abstracts. Use `pdf_url` to fetch the full PDF if needed.

### Wikipedia Reader

Each document contains:

- **Text**: Article title + summary/intro section
- **Metadata**:
  - `title` - Article title
  - `url` - Link to Wikipedia page

**Note**: Only article summaries, not full articles.

## Installation

```bash
pip install llama-index-readers-builtsimple
```

## Quick Start

### Basic Usage

```python
from llama_index.readers.builtsimple import (
    BuiltSimplePubMedReader,
    BuiltSimpleArxivReader,
)

# Search PubMed for medical literature
pubmed_reader = BuiltSimplePubMedReader()
pubmed_docs = pubmed_reader.load_data("CRISPR gene therapy", limit=10)

for doc in pubmed_docs:
    print(f"Title: {doc.metadata['title']}")
    print(f"Journal: {doc.metadata['journal']}")
    print(f"Year: {doc.metadata['pub_year']}")
    print(f"URL: {doc.metadata['url']}\n")

# Search ArXiv for ML papers
arxiv_reader = BuiltSimpleArxivReader()
arxiv_docs = arxiv_reader.load_data(
    "transformer architecture attention", limit=10
)

for doc in arxiv_docs:
    print(f"Title: {doc.metadata['title']}")
    print(f"Authors: {doc.metadata['authors']}")
    print(f"ArXiv ID: {doc.metadata['arxiv_id']}\n")
```

### Build a RAG Index

```python
from llama_index.core import VectorStoreIndex
from llama_index.readers.builtsimple import BuiltSimplePubMedReader

# Load documents
reader = BuiltSimplePubMedReader()
documents = reader.load_data("immunotherapy cancer treatment", limit=20)

# Build index
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What are the side effects of CAR-T therapy?")
print(response)
```

### Combine Multiple Sources

```python
from llama_index.core import VectorStoreIndex
from llama_index.readers.builtsimple import (
    BuiltSimplePubMedReader,
    BuiltSimpleArxivReader,
)

# Load from multiple sources
pubmed = BuiltSimplePubMedReader()
arxiv = BuiltSimpleArxivReader()

# Combine documents
documents = []
documents.extend(pubmed.load_data("drug discovery machine learning", limit=10))
documents.extend(arxiv.load_data("drug discovery deep learning", limit=10))

# Build unified index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query(
    "How is machine learning being used for drug discovery?"
)
print(response)
```

## API Reference

### BuiltSimplePubMedReader

**Constructor Parameters:**

- `api_key` (optional): API key for higher rate limits
- `timeout`: Request timeout in seconds (default: 30)

**load_data() Parameters:**

- `query`: Search query string
- `limit`: Maximum results to return (default: 10)

**Document Metadata:**

- `source`: "builtsimple-pubmed"
- `pmid`: PubMed ID
- `title`: Paper title
- `journal`: Journal name
- `pub_year`: Publication year
- `doi`: DOI identifier
- `url`: Link to PubMed

### BuiltSimpleArxivReader

**Constructor Parameters:**

- `api_key` (optional): API key for higher rate limits
- `timeout`: Request timeout in seconds (default: 30)

**load_data() Parameters:**

- `query`: Search query string
- `limit`: Maximum results to return (default: 10)

**Document Metadata:**

- `source`: "builtsimple-arxiv"
- `arxiv_id`: ArXiv identifier (e.g., "2301.12345")
- `title`: Paper title
- `authors`: Author list
- `year`: Publication year
- `url`: Link to ArXiv

### BuiltSimpleWikipediaReader

**Constructor Parameters:**

- `api_key` (optional): API key for higher rate limits
- `timeout`: Request timeout in seconds (default: 30)

**load_data() Parameters:**

- `query`: Search query string
- `limit`: Maximum results to return (default: 10)

**Document Metadata:**

- `source`: "builtsimple-wikipedia"
- `title`: Article title
- `url`: Link to Wikipedia

## Rate Limits

| Tier | Rate Limit       | Notes             |
| ---- | ---------------- | ----------------- |
| Free | 100 req/month    | No API key needed |
| Pro  | 10,000 req/month | Requires API key  |

Get an API key at [built-simple.ai](https://built-simple.ai).

## Why Built-Simple?

Unlike scraping or official APIs:

- **Pre-indexed vectors** - No embedding costs, instant semantic search
- **Hybrid search** - Combines BM25 + vector similarity
- **Always available** - No rate limit hell from upstream providers
- **Structured data** - Clean JSON responses with full metadata

## Contributing

This package is part of the LlamaIndex ecosystem. To contribute:

1. Fork the repo
2. Create a feature branch
3. Submit a PR to [run-llama/llama_index](https://github.com/run-llama/llama_index)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [Built-Simple](https://built-simple.ai)
- [PubMed API Docs](https://pubmed.built-simple.ai/docs)
- [ArXiv API Docs](https://arxiv.built-simple.ai/docs)
- [LlamaIndex](https://llamaindex.ai)
- [LlamaHub](https://llamahub.ai)
