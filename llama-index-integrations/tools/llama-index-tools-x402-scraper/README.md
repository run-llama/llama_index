# LlamaIndex Tool - x402 Scraper

This tool fetches any URL and converts it to clean, minimal Markdown, designed specifically for RAG pipelines. It includes inherent protection against rate limits by leveraging the emerging x402 machine-to-machine payment standard.

## Usage

```python
from llama_index.tools.x402_scraper import X402ScraperToolSpec

tool_spec = X402ScraperToolSpec()
print(tool_spec.scrape_url("https://example.com"))
```
