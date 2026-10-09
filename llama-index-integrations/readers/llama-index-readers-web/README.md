# llama-index-readers-web

Web page readers for [LlamaIndex](https://github.com/run-llama/llama_index),
bundled in a single package:

```bash
pip install llama-index-readers-web
```

```python
from llama_index.readers.web import TrafilaturaWebReader

documents = TrafilaturaWebReader().load_data(urls=["https://example.com"])
```

See the per-reader READMEs under `llama_index/readers/web/` for the
options each reader supports.

## DomPrunerWebReader

Query-aware, token-budget reader for RAG pipelines. It extracts clean
content from a page (SSR/SSG via DOM AST, `__NEXT_DATA__` payloads, UA
rotation on 403/429, Playwright fallback for CSR) and, when a query is
given, keeps only the BM25-relevant sections within a token budget, with
each section's heading chain preserved in `metadata["heading_path"]`. No
paid API or API key required.

```python
from llama_index.readers.web import DomPrunerWebReader

loader = DomPrunerWebReader(
    query="how does fetch handle redirects",
    token_budget=1200,
)
documents = loader.load_data(urls=["https://example.com/docs"])
```

See `llama_index/readers/web/dompruner_web/README.md` for details.
