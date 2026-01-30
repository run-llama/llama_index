---
title: TzafonWebReader
description: Tzafon is a platform for programmatic control of browsers and desktops.
icon: browser
mode: "wide"
---

# `TzafonWebReader`

## Description

The `TzafonWebReader` enables web scraping and automation using [Tzafon](https://tzafon.ai), a platform for programmatic control of browsers and desktops. This tool allows agents to navigate websites, interact with content, and extract data using Tzafon's stealth and speed capabilities.

Key Features:

- **Full Stealth**: Bypass anti-bot measures with built-in stealth features.
- **Lightning Fast**: Optimized for speed and performance.
- **Multi-Tab Control**: Manage multiple browser tabs programmatically.
- **Page Context API**: Access detailed context about the page structure and content.

## Installation

To use this tool, you need to install the Tzafon SDK along with `playwright`:

```bash
pip install tzafon playwright
```

Get the TZAFON_API_KEY from https://tzafon.ai/dashboard.
The API Key can be passed directly to the reader or set as TZAFON_API_KEY environment variable

## Usage

## Example

The following example demonstrates how to initialize the tool and use it to load a website:

```python Code
from llama_index.readers.web import TzafonWebReader

reader = TzafonWebReader(api_key="your_api_key_here")
docs = reader.load_data(
    urls=["https://example.com"],
    # Return only text content of the webpages. True by Default
    text_content=False,
)
docs
```

## Arguments

The `TzafonWebReader` accepts the following parameters:

| Argument    | Type     | Description                                                           |
| :---------- | :------- | :-------------------------------------------------------------------- |
| **api_key** | `string` | _Optional_. Tzafon API key. Default is `TZAFON_API_KEY` env variable. |

## Return Format

The tool returns the content of the loaded page, typically in a text or html format suitable for LLM processing.
