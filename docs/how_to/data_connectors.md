# Data Connectors (LlamaHub 🦙)

Our data connectors are offered through [LlamaHub](https://llamahub.ai/) 🦙. 
LlamaHub is an open-source repository containing data loaders that you can easily plug and play into any GPT Index application.

![](/_static/data_connectors/llamahub.png)


Some sample data connectors:
- local file directory (`SimpleDirectoryReader`). Can support parsing a wide range of file types: `.pdf`, `.jpg`, `.png`, `.docx`, etc.
- [Notion](https://developers.notion.com/) (`NotionPageReader`)
- [Google Docs](https://developers.google.com/docs/api) (`GoogleDocsReader`)
- [Slack](https://api.slack.com/) (`SlackReader`)
- [Discord](https://discord.com/developers/docs/intro) (`DiscordReader`)


Each data loader contains a "Usage" section showing how that loader can be used. At the core of using each loader is a `download_loader` function, which
downloads the loader file into a module that you can use within your application.

Example usage:

```python
from gpt_index import GPTSimpleVectorIndex, download_loader

GoogleDocsReader = download_loader('GoogleDocsReader')

gdoc_ids = ['1wf-y2pd9C878Oh-FmLH7Q_BQkljdm6TQal-c1pUfrec']
loader = GoogleDocsReader()
documents = loader.load_data(document_ids=gdoc_ids)
index = GPTSimpleVectorIndex(documents)
index.query('Where did the author go to school?')
```

