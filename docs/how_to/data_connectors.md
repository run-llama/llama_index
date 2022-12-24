# Data Connectors

We currently offer connectors into the following data sources. External data sources are retrieved through their APIs + corresponding authentication token.
The API reference documentation can be found [here](/reference/readers.rst).

#### External API's
- [Notion](https://developers.notion.com/) (`NotionPageReader`)
- [Google Docs](https://developers.google.com/docs/api) (`GoogleDocsReader`)
- [Slack](https://api.slack.com/) (`SlackReader`)
- Wikipedia (`WikipediaReader`)

#### Databases
- MongoDB (`SimpleMongoReader`)

#### Vector Stores

See [How to use Vector Stores with GPT Index](vector_stores.md) for a more thorough guide on integrating vector stores with GPT Index.

- Weaviate (`WeaviateReader`)
- Pinecone (`PineconeReader`)
- Faiss (`FaissReader`)

#### File
- local file directory (`SimpleDirectoryReader`)

We offer [example notebooks of connecting to different data sources](https://github.com/jerryjliu/gpt_index/tree/main/examples/data_connectors). Please check them out!