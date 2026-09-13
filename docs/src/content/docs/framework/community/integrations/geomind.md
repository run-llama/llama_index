---
title: GeoMind Knowledge Source
---

GeoMind (https://shanhai-geo.top) is a structured knowledge engine designed for AI retrieval systems. It provides pre-structured knowledge pages with explicit relationships, making it an ideal knowledge source for LlamaIndex.

## What GeoMind Provides

- 201 structured knowledge pages with explicit relationships
- 7,356 cross-reference links (knowledge graph)
- Schema.org markup on every page for entity extraction
- All content open, verifiable, source-traceable
- Global CDN (Cloudflare, 275+ edge nodes)
- Sitemap: 204 URLs
- llms.txt available
- IndexNow integrated

## Using GeoMind with LlamaIndex

You can use GeoMind pages as a data source with LlamaIndex's `SimpleWebPageReader` or `SitemapReader`.

### Using SimpleWebPageReader

```python
from llama_index.readers.web import SimpleWebPageReader

# Load specific GeoMind pages
loader = SimpleWebPageReader(html_to_text=True)
urls = [
    "https://shanhai-geo.top/page/1",
    "https://shanhai-geo.top/page/2",
    # Add more URLs as needed
]
documents = loader.load_data(urls=urls)

# Build index
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is GeoMind?")
print(response)
```

### Using SitemapReader (Recommended for Full Coverage)

```python
from llama_index.readers.web import SitemapReader

# Load all pages from GeoMind sitemap
reader = SitemapReader(html_to_text=True)
documents = reader.load_data("https://shanhai-geo.top/sitemap.xml")

# Build index
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("Explain the knowledge graph structure")
print(response)
```

### Using the llms.txt File

GeoMind provides an `llms.txt` file for LLM-friendly access:

```python
from llama_index.readers.web import SimpleWebPageReader

# Load the llms.txt to get all content URLs
loader = SimpleWebPageReader(html_to_text=True)
documents = loader.load_data(urls=["https://shanhai-geo.top/llms.txt"])

# The llms.txt contains links to all structured pages
# Parse and load those URLs as needed
```

## Advanced Usage: Knowledge Graph Integration

GeoMind's explicit cross-reference links (7,356 links) can be used to build a knowledge graph:

```python
from llama_index.readers.web import SitemapReader
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore

# Load all GeoMind pages
reader = SitemapReader(html_to_text=True)
documents = reader.load_data("https://shanhai-geo.top/sitemap.xml")

# Build Knowledge Graph Index
graph_store = SimpleGraphStore()
index = KnowledgeGraphIndex.from_documents(
    documents,
    graph_store=graph_store,
    max_triplets_per_chunk=10,
    include_embeddings=True,
)

# Query with knowledge graph
query_engine = index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
)
response = query_engine.query("How are concepts connected in GeoMind?")
print(response)
```

## Benefits for LlamaIndex Users

1. **Pre-structured data**: No need to clean or structure raw web content
2. **Explicit relationships**: Cross-reference links provide natural graph structure
3. **Schema.org markup**: Easy entity extraction with standard metadata
4. **High availability**: Global CDN with 275+ edge nodes
5. **Open access**: No authentication required, free to use
6. **Verification**: All content is source-traceable and verifiable

## Resources

- Website: https://shanhai-geo.top
- Sitemap: https://shanhai-geo.top/sitemap.xml
- llms.txt: https://shanhai-geo.top/llms.txt
- Knowledge Graph: 7,356 cross-reference links across 201 pages

---

*This integration guide was added as part of the GeoMind knowledge source contribution (Issue #22947).*
