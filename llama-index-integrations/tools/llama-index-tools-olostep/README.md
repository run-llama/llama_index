# Olostep Tool

[Olostep](https://olostep.com/) is a web scraping, crawling, and research API designed for agents. It enables LLM agents to extract data from websites, search the web, and synthesize answers from web sources.

To begin, you need to obtain an API key on the [Olostep dashboard](https://dashboard.olostep.com/).

## Installation

```bash
pip install llama-index-tools-olostep
```

## Usage

Here's a basic example of using the OlostepToolSpec with a LlamaIndex agent:

```python
from llama_index.tools.olostep import OlostepToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = OlostepToolSpec(api_key="your-olostep-api-key")
agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
)

# Scrape a URL and summarize it
await agent.run("Scrape https://example.com and summarize the main content")
```

## Use Case 1: Basic Scraping

Scrape a single URL and extract its content:

```python
from llama_index.tools.olostep import OlostepToolSpec

tool_spec = OlostepToolSpec(api_key="your-api-key")

# Scrape a URL
documents = tool_spec.scrape_url(
    url="https://example.com",
    formats="markdown",
    wait_before_scraping=2000,  # Wait 2 seconds for JS rendering
)

for doc in documents:
    print(f"URL: {doc.extra_info['url']}")
    print(f"Content: {doc.text[:500]}...")
```

## Use Case 2: Research Agent

Use search and answer questions to conduct web research:

```python
from llama_index.tools.olostep import OlostepToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = OlostepToolSpec(api_key="your-api-key")

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
)

# Research task
result = await agent.run(
    "Research the current CEO of OpenAI and return their background information"
)
print(result)
```

## Use Case 3: Website Discovery and Batch Scraping

Map a website's structure, then batch scrape selected pages:

```python
from llama_index.tools.olostep import OlostepToolSpec

tool_spec = OlostepToolSpec(api_key="your-api-key")

# First, discover all URLs on a website
map_docs = tool_spec.map_website(
    url="https://docs.example.com",
    include_urls="/guides/**,/api/**",
)

# Extract URLs from the document
all_urls = map_docs[0].text.split("\n")[:10]  # Get first 10 URLs

# Batch scrape them
batch_docs = tool_spec.batch_scrape(
    urls=",".join(all_urls),
    formats="markdown",
)

print(f"Scraped {len(batch_docs)} pages")
```

## Available Tools

| Tool              | Description                                                                 |
| ----------------- | --------------------------------------------------------------------------- |
| `scrape_url`      | Scrape a single URL and extract its content (markdown, HTML, text, or JSON) |
| `crawl_website`   | Crawl an entire website or section with URL filtering and search queries    |
| `map_website`     | Discover all URLs on a website from sitemaps and links                      |
| `search_web`      | Search the web and return relevant links with titles and descriptions       |
| `answer_question` | Search the web and synthesize an AI-powered answer from verified sources    |
| `batch_scrape`    | Scrape multiple URLs concurrently (most efficient for 50-10,000 URLs)       |

## Advanced Features

### Structured Data Extraction

Extract structured data using pre-built parsers:

```python
tool_spec.scrape_url(
    url="https://amazon.com/dp/B123456789",
    parser_id="@olostep/amazon-it-product",
    formats="json",
)
```

Available parsers:

- `@olostep/google-search` - Extract Google search results
- `@olostep/amazon-it-product` - Extract Amazon product details
- `@olostep/extract-emails` - Extract email addresses from pages
- `@olostep/extract-socials` - Extract social media links

### Geo-location Based Scraping

Scrape pages as if you're accessing from a specific country:

```python
tool_spec.scrape_url(
    url="https://example.com",
    country="gb",  # Scrape as if from UK
)
```

### Website Crawling with Filtering

Crawl a site with include/exclude patterns:

```python
tool_spec.crawl_website(
    url="https://blog.example.com",
    max_pages=100,
    include_urls="/posts/**,/tutorials/**",
    exclude_urls="/admin/**,/draft/**",
    search_query="machine learning",  # Only crawl pages relevant to this
)
```

## Resources

- [Olostep Website](https://olostep.com/)
- [Olostep Documentation](https://docs.olostep.com/)
- [Olostep Python SDK](https://github.com/olostep/python-sdk)
- [API Reference](https://docs.olostep.com/api)

## Additional Examples

A comprehensive Jupyter notebook with more examples is available [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-olostep/examples/olostep.ipynb).
