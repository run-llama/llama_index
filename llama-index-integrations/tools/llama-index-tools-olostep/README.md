# LlamaIndex Tools Integration: Olostep

[Olostep](https://www.olostep.com) is web scraping, crawling, and search
infrastructure built for AI agents. This tool gives a LlamaIndex agent live
web access — scrape any URL to clean markdown, get an AI-grounded answer with
sources, crawl a website, and discover the URLs on a site.

To get started, get an API key from the
[Olostep dashboard](https://www.olostep.com/dashboard/api-keys).

## Installation

```bash
pip install llama-index-tools-olostep
```

## Usage

Set your key as an environment variable (recommended):

```bash
export OLOSTEP_API_KEY="your-key"
```

```python
import os
from llama_index.tools.olostep import OlostepToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

olostep_tool = OlostepToolSpec(api_key=os.environ["OLOSTEP_API_KEY"])

agent = FunctionAgent(
    tools=olostep_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
)

await agent.run("Scrape https://example.com and summarize the main heading")
```

## Available tools

- **`scrape`** — Scrape a single URL and return its content as clean markdown.
- **`answer`** — Get an AI-grounded answer to a question, synthesized from live
  web sources with citations.
- **`crawl`** — Crawl a website from a start URL and return the pages' content.
- **`map_site`** — Discover all the URLs on a website.