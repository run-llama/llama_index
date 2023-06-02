# 🔌 Data Connectors (LlamaHub)

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

Each data loader contains a "Usage" section showing how that loader can be used. At the core of using each loader is a `download_loader` function, which
downloads the loader file into a module that you can use within your application.

Example usage:

```python
from llama_index import GPTVectorStoreIndex, download_loader

GoogleDocsReader = download_loader('GoogleDocsReader')

gdoc_ids = ['1wf-y2pd9C878Oh-FmLH7Q_BQkljdm6TQal-c1pUfrec']
loader = GoogleDocsReader()
documents = loader.load_data(document_ids=gdoc_ids)
index = GPTVectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
query_engine.query('Where did the author go to school?')
```

```{toctree}
---
caption: Examples
maxdepth: 1
---

../examples/data_connectors/PsychicDemo.ipynb
../examples/data_connectors/DeepLakeReader.ipynb
../examples/data_connectors/QdrantDemo.ipynb
../examples/data_connectors/DiscordDemo.ipynb
../examples/data_connectors/MongoDemo.ipynb
../examples/data_connectors/ChromaDemo.ipynb
../examples/data_connectors/MyScaleReaderDemo.ipynb
../examples/data_connectors/FaissDemo.ipynb
../examples/data_connectors/ObsidianReaderDemo.ipynb
../examples/data_connectors/SlackDemo.ipynb
../examples/data_connectors/WebPageDemo.ipynb
../examples/data_connectors/PineconeDemo.ipynb
../examples/data_connectors/MboxReaderDemo.ipynb
../examples/data_connectors/MilvusReaderDemo.ipynb
../examples/data_connectors/NotionDemo.ipynb
../examples/data_connectors/GithubRepositoryReaderDemo.ipynb
../examples/data_connectors/GoogleDocsDemo.ipynb
../examples/data_connectors/DatabaseReaderDemo.ipynb
../examples/data_connectors/TwitterDemo.ipynb
../examples/data_connectors/WeaviateDemo.ipynb
../examples/data_connectors/MakeDemo.ipynb
```

