# ClearFetch Reader

This loader reads content from web URLs using [ClearFetch](https://clearfetch.dev), an unblocked web scraping and Model Context Protocol (MCP) context layer.

ClearFetch bypasses Cloudflare Turnstile, DataDome, and anti-bot gates to return clean, token-optimized Markdown documents suitable for LLMs, RAG pipelines, and AI Agents.

## Installation

```bash
pip install llama-index-readers-clearfetch
```

## Usage

```python
from llama_index.readers.clearfetch import ClearFetchReader

# Initialize reader with your API key (or set CLEARFETCH_API_KEY env variable)
reader = ClearFetchReader(api_key="cf_live_...")

# Load web pages as LlamaIndex Document objects
documents = reader.load_data(
    urls=["https://news.ycombinator.com", "https://example.com"],
    stealth=True
)

for doc in documents:
    print("Title:", doc.extra_info.get("title"))
    print("Content preview:", doc.text[:200])
```

## Features
- **Turnstile & Anti-Bot Auto-Bypass**: Priority residential proxy routing.
- **LLM Token Optimization**: Strips HTML noise, delivering up to 80% fewer tokens.
- **Zero Configuration**: Ready for production out of the box with 500 free credits upon signup at [clearfetch.dev](https://clearfetch.dev).
