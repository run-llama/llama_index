# LlamaHub

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

## Modules

Some sample data connectors:

- local file directory (`SimpleDirectoryReader`). Can support parsing a wide range of file types: `.pdf`, `.jpg`, `.png`, `.docx`, etc.
- [Notion](https://developers.notion.com/) (`NotionPageReader`)
- [Google Docs](https://developers.google.com/docs/api) (`GoogleDocsReader`)
- [Slack](https://api.slack.com/) (`SlackReader`)
- [Discord](https://discord.com/developers/docs/intro) (`DiscordReader`)
- [Apify Actors](https://llamahub.ai/l/apify-actor) (`ApifyActor`). Can crawl the web, scrape webpages, extract text content, download files including `.pdf`, `.jpg`, `.png`, `.docx`, etc.

See below for detailed guides.
