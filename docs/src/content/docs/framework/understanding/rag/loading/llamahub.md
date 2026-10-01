---
title: Finding Data Connectors
---

Data connectors (also called `Reader`s) load data from external sources into LlamaIndex `Document` objects. Beyond the built-in `SimpleDirectoryReader`, every connector is published as its own package so you only install what you need.

## Finding a connector

- Browse the [`readers` directory](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers) of the LlamaIndex repo. Each package has a README with installation and usage instructions.
- Search PyPI for `llama-index-readers-`. Package names follow the pattern `llama-index-readers-<source>`, for example `llama-index-readers-google` or `llama-index-readers-database`.
- The same layout holds for other integration types: agent tools live under [`tools`](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/tools), LLMs under [`llms`](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/llms), embeddings under [`embeddings`](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/embeddings), and so on. See the [integrations overview](/python/framework/community/integrations) for the full picture.

## Usage Pattern

Install the package, then import the reader from the matching namespace:

```bash
pip install llama-index-readers-google
```

```python
from llama_index.readers.google import GoogleDocsReader

loader = GoogleDocsReader()
documents = loader.load_data(document_ids=[...])
```

## Built-in connector: SimpleDirectoryReader

`SimpleDirectoryReader` can parse a wide range of file types including `.md`, `.pdf`, `.jpg`, `.png`, `.docx`, as well as audio and video types. It is available directly as part of LlamaIndex:

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
```

## Popular connectors

A few of the hundreds available:

- [Notion](https://developers.notion.com/) (`NotionPageReader`, `llama-index-readers-notion`)
- [Google Docs](https://developers.google.com/docs/api) (`GoogleDocsReader`, `llama-index-readers-google`)
- [Slack](https://api.slack.com/) (`SlackReader`, `llama-index-readers-slack`)
- [Discord](https://discord.com/developers/docs/intro) (`DiscordReader`, `llama-index-readers-discord`)
- [Apify Actors](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers/llama-index-readers-apify) (`ApifyActor`, `llama-index-readers-apify`). Can crawl the web, scrape webpages, extract text content, download files including `.pdf`, `.jpg`, `.png`, `.docx`, etc.

See the [data connectors module guide](/python/framework/module_guides/loading/connector) for more details.
