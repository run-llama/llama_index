# StripeDocs Loader

```bash
pip install llama-index-readers-stripe-docs
```

This loader asynchronously loads data from the [Stripe documentation](https://stripe.com/docs). It iterates through the Stripe sitemap to get all `/docs` references.

It is based on the [Async Website Loader](https://llamahub.ai/l/web-async_web).

## Usage

```python
from llama_index.core import VectorStoreIndex
from llama_index.readers.stripe_docs import StripeDocsReader

loader = StripeDocsReader()
documents = loader.load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
query_engine.query("How do I accept payments on my website?")
```

The `StripeDocsReader` allows you to return plain text docs by setting `html_to_text=True`. You can also adjust the maximum concurrent requests by setting `limit=10`.

## Filtering

You can filter pages from the Stripe sitemap by adding the _filters_ argument to the load_data method. This allows you to control what pages from the Stripe website, including documentation, will be loaded.

The default filters are set to `["/docs"]` to scope everything to docs only.

```python
documents = loader.load_data(filters=["/terminal"])
```
