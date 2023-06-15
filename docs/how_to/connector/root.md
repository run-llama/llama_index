# 🔌 Data Connectors (LlamaHub)

## Concept
A data connector (i.e. `Reader`) ingest data from different data sources and data formats into a simple `Document` representation (text and simple metadata).

Once you've ingested your data, you can build an [Index](/how_to/index/root.md) on top, ask questions using a [Query Engine](/how_to/query_engine/root.md), and have a conversation using a [Chat Engine](/how_to/chat_engine/root.md).
## LlamaHub
Our data connectors are offered through [LlamaHub](https://llamahub.ai/) 🦙. 
LlamaHub is an open-source repository containing data loaders that you can easily plug and play into any LlamaIndex application.

![](/_static/data_connectors/llamahub.png)


## Usage Pattern
Get started with:
```python
from llama_index import download_loader

GoogleDocsReader = download_loader('GoogleDocsReader')
loader = GoogleDocsReader()
documents = loader.load_data(document_ids=[...])
```

```{toctree}
---
maxdepth: 2
---
usage_pattern.md
```


## Modules

Some sample data connectors:
- local file directory (`SimpleDirectoryReader`). Can support parsing a wide range of file types: `.pdf`, `.jpg`, `.png`, `.docx`, etc.
- [Notion](https://developers.notion.com/) (`NotionPageReader`)
- [Google Docs](https://developers.google.com/docs/api) (`GoogleDocsReader`)
- [Slack](https://api.slack.com/) (`SlackReader`)
- [Discord](https://discord.com/developers/docs/intro) (`DiscordReader`)
- [Apify Actors](https://llamahub.ai/l/apify-actor) (`ApifyActor`). Can crawl the web, scrape webpages, extract text content, download files including `.pdf`, `.jpg`, `.png`, `.docx`, etc.

See below for detailed guides.

```{toctree}
---
maxdepth: 2
---
modules.rst
```