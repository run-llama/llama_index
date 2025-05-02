# Playwright Browser Tool

This tool is a wrapper around the Playwright library. It allows you to navigate to a website, extract text and hyperlinks, and click on elements.

> **Warning**
> Only support async functions and playwright browser APIs.

## Installation

```
pip install llama-index-tools-playwright
```

## Setup

In order to use this tool, you need to have a async Playwright browser instance. You can hook one up by running the following code:

```python
browser = PlaywrightToolSpec.create_async_playwright_browser(headless=False)
playwright_tool = PlaywrightToolSpec.from_async_browser(browser)
```

## Usage

### Navigate to a website

```python
await playwright_tool.navigate_to("https://playwright.dev/python/docs/intro")
```

### Navigate back

```python
await playwright_tool.navigate_back()
```

### Get current page URL

```python
await playwright_tool.get_current_page()
```

### Extract all hyperlinks

```python
await playwright_tool.extract_hyperlinks()
```

### Extract all text

```python
await playwright_tool.extract_text()
```

### Get element attributes

```python
element = await playwright_tool.get_elements(
    selector="ELEMENT_SELECTOR", attributes=["innerText"]
)
```

### Click on an element

```python
await playwright_tool.click(selector="ELEMENT_SELECTOR")
```

### Fill in an input field

```python
await playwright_tool.fill(selector="ELEMENT_SELECTOR", value="Hello")
```

## Agentic Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-playwright/examples/playwright_browser_agent.ipynb)
