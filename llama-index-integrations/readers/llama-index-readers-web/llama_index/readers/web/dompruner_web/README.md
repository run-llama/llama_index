# DomPruner Web Reader

```bash
pip install llama-index-readers-web
```

Query-aware, token-budget web page reader for RAG pipelines. It fetches
pages through [dompruner](https://pypi.org/project/dompruner)'s
zero-dependency pipeline and returns only the sections you need, so a
15,000-token documentation page becomes a few hundred tokens instead of
exhausting your context budget before retrieval even begins. No paid API
and no API key required.

Processing happens in two separate stages:

1. **Extraction**: deterministic content extraction over the DOM AST that
   strips page furniture (nav, footer, cookie banners). SSR and static
   pages (Docusaurus, MkDocs, Next.js) are handled with a stdlib parser,
   `__NEXT_DATA__` payloads are read directly when present, and pages
   that return 403/429 are retried with rotated user agents before a
   Playwright fallback is attempted for true CSR pages.
2. **Pruning**: when a `query` is given, BM25 scores the sections of the
   clean content and only the relevant sections within `token_budget`
   are kept. Fenced code blocks stay whole, so code signatures are never
   split, and every returned document keeps its ancestor heading chain
   (H1 -> H2 -> H3) in `metadata["heading_path"]`.

## Usage

```python
from llama_index.readers.web import DomPrunerWebReader

loader = DomPrunerWebReader(
    query="how does fetch handle redirects",
    token_budget=1200,
)
documents = loader.load_data(
    urls=["https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API"],
)
for doc in documents:
    print(doc.metadata["heading_path"], doc.metadata["section_title"])
```

Each returned document is one heading-delimited section:

```python
{
    "url": "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API",
    "section_title": "Redirect modes",
    "heading_path": ["Fetch API", "Redirect modes"],
    "render_type": "SSR",
    "original_tokens": 15965,
    "refined_tokens": 1368,
    "reduction_ratio": 0.914,
    "bm25_confidence": 4.2,
    "title": "Fetch API - MDN",
    "lang": "en",
}
```

### Additional Parameters

```python
# Without a query the full extracted content is returned,
# split into sections but never pruned.
loader = DomPrunerWebReader()

# Disable the token cap entirely.
loader = DomPrunerWebReader(query="usage", token_budget=None)
```

Note that BM25 selection itself runs at dompruner's internal budget
(approximately 1,200 tokens): `token_budget` is a hard cap applied on top,
so it guarantees the output never exceeds the cap rather than filling it
up to it. A single section larger than the cap is kept whole instead of
being split mid-signature.

Client-side rendered pages fall back to Playwright automatically; run
`playwright install chromium` once if the page needs a real browser.

## Examples

### LlamaIndex

```python
from llama_index.core import VectorStoreIndex
from llama_index.readers.web import DomPrunerWebReader

loader = DomPrunerWebReader(
    query="How do I paginate results?",
    token_budget=1000,
)
documents = loader.load_data(urls=["https://example.com/api/docs"])
index = VectorStoreIndex.from_documents(documents)
index.as_query_engine().query("How do I paginate results?")
```
