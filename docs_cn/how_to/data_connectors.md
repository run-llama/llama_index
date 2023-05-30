我们的数据连接器通过[LlamaHub](https://llamahub.ai/) 🦙提供。LlamaHub是一个开源存储库，其中包含您可以轻松插入任何LlamaIndex应用程序的数据加载程序。

一些示例数据连接器：
- 本地文件目录（`SimpleDirectoryReader`）。可以支持解析各种文件类型：`.pdf`，`.jpg`，`.png`，`.docx`等。
- [Notion](https://developers.notion.com/)（`NotionPageReader`）
- [Google Docs](https://developers.google.com/docs/api)（`GoogleDocsReader`）
- [Slack](https://api.slack.com/)（`SlackReader`）
- [Discord](https://discord.com/developers/docs/intro)（`DiscordReader`）
- [Apify Actors](https://llamahub.ai/l/apify-actor)（`ApifyActor`）。可以爬取网络，抓取网页，提取文本内容，下载文件，包括`.pdf`，`.jpg`，`.png`，`.docx`等。

每个数据加载程序都包含一个“使用”部分，显示如何使用该加载程序。使用每个加载程序的核心是一个`download_loader`函数，它将加载程序文件下载到您可以在应用程序中使用的模块中。

示例用法：

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

我们的数据连接器通过[LlamaHub](https://llamahub.ai/) 🦙提供。LlamaHub是一个开源存储库，其中包含您可以轻松插入任何LlamaIndex应用程序的数据加载程序。一些示例数据连接器包括本地文件目录（`SimpleDirectoryReader`）、[Notion](https://developers.notion.com/)（`NotionPageReader`）、[Google Docs](https://developers.google.com/docs/api)（`GoogleDocsReader`）、[Slack](https://api.slack.com/)（`SlackReader`）、[Discord](https://discord.com/developers/docs/intro)（`DiscordReader`）和[Apify Actors](https://llamahub.ai/l/apify-actor)（`ApifyActor`）。每个数据加载程序都包含一个“使用”部分，显示如何使用该加载程序。使用每个加载程序的核心是一个`download_loader`函数，它将加载程序文件下载到您可以在应用程序中使用的模块中。