# LlamaHub

Our data connectors are offered through [LlamaHub](https://llamahub.ai/) 🦙.
LlamaHub contains a registry of open-source data connectors that you can easily plug into any LlamaIndex application (+ Agent Tools, and Llama Packs).

![](/_static/data_connectors/llamahub.png)

## Usage Pattern

Get started with:

```python
from llama_index import download_loader

GoogleDocsReader = download_loader("GoogleDocsReader")
loader = GoogleDocsReader()
documents = loader.load_data(document_ids=[...])
```

## Built-in connector: SimpleDirectoryReader

`SimpleDirectoryReader`. Can support parsing a wide range of file types including `.md`, `.pdf`, `.jpg`, `.png`, `.docx`, as well as audio and video types. It is available directly as part of LlamaIndex:

```python
from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
```

## Available connectors

Browse [LlamaHub](https://llamahub.ai/) directly to see the hundreds of connectors available, including:

- [Notion](https://developers.notion.com/) (`NotionPageReader`)
- [Google Docs](https://developers.google.com/docs/api) (`GoogleDocsReader`)
- [Slack](https://api.slack.com/) (`SlackReader`)
- [Discord](https://discord.com/developers/docs/intro) (`DiscordReader`)
- [Apify Actors](https://llamahub.ai/l/apify-actor) (`ApifyActor`). Can crawl the web, scrape webpages, extract text content, download files including `.pdf`, `.jpg`, `.png`, `.docx`, etc.
