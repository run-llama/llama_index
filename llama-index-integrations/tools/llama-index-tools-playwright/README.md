# Playwright Browser Tool

This tool is a wrapper around the Playwright library. It allows you to navigate to a website, extract text and hyperlinks, and click on elements.

## Installation

```
pip install llama-index-tools-playwright
```

## Setup

In order to use this tool, you need to have a sync Playwright browser instance. You can hook one up by running the following code:

```python
browser = PlaywrightToolSpec.create_sync_playwright_browser(headless=False)
playwright_tool = PlaywrightToolSpec.from_sync_browser(browser)
```

## Usage

### Navigate to a website

```python
playwright_tool.navigate_to("https://playwright.dev/python/docs/intro")
```

### Get current page URL

```python
playwright_tool.get_current_page()
```

### Extract all hyperlinks

```python
playwright_tool.extract_hyperlinks()
```

### Extract all text

```python
playwright_tool.extract_text()
```

### Get element attributes

```python
element = playwright_tool.get_elements(
    selector="ELEMENT_SELECTOR", attributes=["innerText"]
)
```

### Click on an element

```python
playwright_tool.click(selector="ELEMENT_SELECTOR")
```

### Fill in an input field

```python
playwright_tool.fill(selector="ELEMENT_SELECTOR", value="Hello")
```
