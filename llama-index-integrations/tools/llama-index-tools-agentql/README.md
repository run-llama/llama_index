# llama-index-tools-agentql

[AgentQL](https://www.agentql.com/) provides web interaction and structured data extraction from any web page using an [AgentQL query](https://docs.agentql.com/agentql-query) or a Natural Language prompt. AgentQL can be used across multiple languages and web pages without breaking over time and change.

> **Warning**
> Only supports async functions and playwright browser APIs, please refer to the following PR for more details: https://github.com/run-llama/llama_index/pull/17808

## Installation

```bash
pip install llama-index-tools-agentql
```

You also need to configure the `AGENTQL_API_KEY` environment variable. You can acquire an API key from our [Dev Portal](https://dev.agentql.com).

## Overview

AgentQL provides the following three function tools:

- **`extract_web_data_with_rest_api`**: Extracts structured data as JSON from a web page given a URL using either an [AgentQL query](https://docs.agentql.com/agentql-query/query-intro) or a Natural Language description of the data.

- **`extract_web_data_from_browser`**: Extracts structured data as JSON from the active web page in a browser using either an [AgentQL query](https://docs.agentql.com/agentql-query/query-intro) or a Natural Language description. **This tool must be used with a Playwright browser.**

- **`get_web_element_from_browser`**: Finds a web element on the active web page in a browser using a Natural Language description and returns its CSS selector for further interaction. **This tool must be used with a Playwright browser.**

You can learn more about how to use AgentQL tools in this [Jupyter notebook](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-agentql/examples/AgentQL_browser_agent.ipynb).

### Extract data using REST API

```python
from llama_index.tools.agentql import AgentQLRestAPIToolSpec

agentql_rest_api_tool = AgentQLRestAPIToolSpec()
await agentql_rest_api_tool.extract_web_data_with_rest_api(
    url="https://www.agentql.com/blog",
    query="{ posts[] { title url author date }}",
)
```

### Work with data and web elements using browser

#### Setup

In order to use the `extract_web_data_from_browser` and `get_web_element_from_browser`, you need to have a Playwright browser instance. If you do not have an active instance, you can initiate one using the `create_async_playwright_browser` utility method from LlamaIndex's Playwright ToolSpec.

> **Note**
> AgentQL browser tools are best used along with LlamaIndex's [Playwright tools](https://docs.llamaindex.ai/en/stable/api_reference/tools/playwright/).

```python
from llama_index.tools.playwright.base import PlaywrightToolSpec

async_browser = await PlaywrightToolSpec.create_async_playwright_browser()
```

You can also use an existing browser instance via Chrome DevTools Protocol (CDP) connection URL:

```python
p = await async_playwright().start()
async_browser = await p.chromium.connect_over_cdp("CDP_CONNECTION_URL")
```

#### Extract data from the active browser page

```python
from llama_index.tools.agentql import AgentQLBrowserToolSpec

playwright_tool = PlaywrightToolSpec(async_browser=async_browser)
await playwright_tool.navigate_to("https://www.agentql.com/blog")

agentql_browser_tool = AgentQLBrowserToolSpec(async_browser=async_browser)
await agentql_browser_tool.extract_web_data_from_browser(
    prompt="the blog posts with title and url",
)
```

#### Find a web element on the active browser page

```python
next_page_button = await agentql_browser_tool.get_web_element_from_browser(
    prompt="The next page navigation button",
)

await playwright_tool.click(next_page_button)
```

## Agentic Usage

This tool has a more extensive example for agentic usage documented in this [Jupyter notebook](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-agentql/examples/AgentQL_browser_agent.ipynb).

## Run tests

In order to run integration tests, you need to configure LLM credentials by setting the `OPENAI_API_KEY` and `AGENTQL_API_KEY` environment variables first. Then run the tests with the following command:

```bash
make test
```
