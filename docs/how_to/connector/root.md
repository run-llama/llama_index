# 🔌 Data Connectors (LlamaHub)

# Concept
A data connector (i.e. `Reader`) ingest data from different data sources and data formats into a simple `Document` representation (text and simple metadata).

Our data connectors are offered through [LlamaHub](https://llamahub.ai/) 🦙. 
LlamaHub is an open-source repository containing data loaders that you can easily plug and play into any LlamaIndex application.

![](/_static/data_connectors/llamahub.png)


Some sample data connectors:
- local file directory (`SimpleDirectoryReader`). Can support parsing a wide range of file types: `.pdf`, `.jpg`, `.png`, `.docx`, etc.
- [Notion](https://developers.notion.com/) (`NotionPageReader`)
- [Google Docs](https://developers.google.com/docs/api) (`GoogleDocsReader`)
- [Slack](https://api.slack.com/) (`SlackReader`)
- [Discord](https://discord.com/developers/docs/intro) (`DiscordReader`)
- [Apify Actors](https://llamahub.ai/l/apify-actor) (`ApifyActor`). Can crawl the web, scrape webpages, extract text content, download files including `.pdf`, `.jpg`, `.png`, `.docx`, etc.


## Usage Pattern

```{toctree}
---
maxdepth: 2
---
usage_pattern.md
```


## Modules
```{toctree}
---
maxdepth: 2
---
modules.md
```