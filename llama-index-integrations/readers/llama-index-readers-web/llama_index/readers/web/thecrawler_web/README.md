# TheCrawler Web Loader

This loader fetches one or more URLs through the [TheCrawler](https://github.com/manchittlab/TheCrawler) hosted API and returns LlamaIndex `Document` objects.

TheCrawler returns boilerplate-stripped markdown plus rich metadata (title, description, status code, structured error type, response timing, etc.) per page. PDF and DOCX URLs are auto-handled by the server, which makes the reader suitable as a generic RAG ingestion step.

## Setup

```bash
pip install requests
```

You also need an API key. Get one at [miaibot.ai](https://www.miaibot.ai) (keys look like `mai_live_<32_hex_chars>`). You can pass it directly or set the `THECRAWLER_API_KEY` environment variable.

## Usage

```python
from llama_index.readers.web import TheCrawlerWebReader
from llama_index.core import SummaryIndex

reader = TheCrawlerWebReader(api_key="mai_live_...")

documents = reader.load_data(
    urls=[
        "https://en.wikipedia.org/wiki/LLamaIndex",
        "https://www.python.org/",
    ]
)

index = SummaryIndex.from_documents(documents)
query_engine = index.as_query_engine()
print(query_engine.query("What is LlamaIndex?"))
```

### Forwarding extra options

Any extra option supported by the TheCrawler API can be passed via `params` and is forwarded to the request body:

```python
reader = TheCrawlerWebReader(
    api_key="mai_live_...",
    params={"usePlaywright": True, "requestTimeoutSecs": 60},
)
```

### Self-hosted instance

To point the reader at a self-hosted `thecrawler-api` server instead of the hosted API:

```python
reader = TheCrawlerWebReader(
    api_key="your-self-hosted-secret",
    api_url="http://localhost:3000/v1",
)
```

### Error handling

The reader never throws for an individual failing URL. Failed pages are returned as `Document` instances with empty text and structured error details in `metadata`:

```python
for doc in reader.load_data(urls=["https://does-not-exist.invalid"]):
    if doc.metadata.get("status") == "error":
        print(
            doc.metadata.get("error_type"),  # e.g. "dns"
            doc.metadata.get("error_retryable"),  # bool
            doc.metadata.get("error"),  # human-readable
        )
```
