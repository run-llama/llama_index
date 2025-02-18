# AgentQL Tool

[AgentQL](https://www.agentql.com/) is a tool for web agents to interact with web elements and extracting data from web pages using Natural language or [AgentQL query](https://docs.agentql.com/agentql-query) for more precise action.

> **Warning**
> Only support async functions and playwright browser APIs.

## Installation

```bash
pip install llama-index-tools-agentql
```

And you should configure credentials by setting the following environment variables:

- AGENTQL_API_KEY

## Overview

AgentQL provide the following three tools:

- **`extract_web_data`**: Extract structured data in JSON form from webpage provided by a url with either natural language description or AgentQL query.

- **`extract_web_data_with_browser`**: Extracted structured data in JSON from the current browser webpage with either natural language description or AgentQL query. (Must use with a browser instance)

- **`extract_web_element_with_browser`**: Extract the CSS selector of a web element from the current browser webpage with natural language description. (Must use with a browser instance)

## Setup

In order to use this tool, you need to have a async Playwright browser instance. You can hook one up using the `create_async_playwright_browser` method:

```python
browser = await AgentQLToolSpec.create_async_playwright_browser(headless=False)
agentql_tool = AgentQLToolSpec.from_async_browser(browser)
```

## Usage

### Extract data from a webpage

```python
await agentql_tool.extract_web_data(
    "https://www.agentql.com/blog",
    query="{ blogs[] { title url author date }}",
)
```

### Extract data from the current browser instance

> **Note**
> Agentql browser tools are best used along with [playwright tools](https://llamahub.ai/l/tools/llama-index-tools-playwright?from=).

```python
await playwright_tool.navigate_to("https://www.agentql.com/blog")
await agentql_tool.extract_web_data_with_browser(
    prompt="Extract all the blog titles and urls from the current page.",
)
```

### Extract the CSS selector of a web element from the current browser instance

```python
next_page_button = await agentql_tool.extract_web_element_with_browser(
    prompt="Button to navigate to the next blog page.",
)
await playwright_tool.click(next_page_button)
```

## Agentic Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-agentql/examples/AgentQL_browser_agent.ipynb)
