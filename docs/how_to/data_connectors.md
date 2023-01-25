# Data Connectors

We currently offer connectors into the following data sources. External data sources are retrieved through their APIs + corresponding authentication token.
The API reference documentation can be found [here](/reference/readers.rst).

All readers can be imported through `from gpt_index.readers import ...`. A subset can be imported directly through `from gpt_index import ...`

#### External API's

- [Notion](https://developers.notion.com/) (`NotionPageReader`)
- [Google Docs](https://developers.google.com/docs/api) (`GoogleDocsReader`)
- [Slack](https://api.slack.com/) (`SlackReader`)
- [Discord](https://discord.com/developers/docs/intro) (`DiscordReader`)
  - Note: We use the [discord.py](https://github.com/Rapptz/discord.py) API wrapper for Discord. This is meant to be used
    in an async setting; however, we adapt it to synchronous Document loading.
- Wikipedia (`WikipediaReader`)
- YouTube (`YoutubeTranscriptReader`)
- Twitter (`TwitterTweetReader`)
- Web (`SimpleWebPageReader`, `BeautifulSoupWebReader`, `TrafilaturaWebReader`)

#### Databases

- MongoDB (`SimpleMongoReader`)
- SQL Databases (`DatabaseReader`)

#### Vector Stores

See [How to use Vector Stores with GPT Index](vector_stores.md) for a more thorough guide on integrating vector stores with GPT Index.

- Weaviate (`WeaviateReader`)
- Pinecone (`PineconeReader`)
- Faiss (`FaissReader`)

### Workflow Automation

- Make.com (`MakeWrapper`). NOTE: `load_data` is not supported. See `pass_response_to_webhook` in the [reference documentation](/reference/readers.rst) instead.

#### File

- local file directory (`SimpleDirectoryReader`)

The `SimpleDirectoryReader` can support parsing a wide range of file types: `.pdf`, `.jpg`, `.png`, `.docx`, `.mp3`, `.mp4`.
Each of these file types may require additional dependencies to be installed.

We offer [example notebooks of connecting to different data sources](https://github.com/jerryjliu/gpt_index/tree/main/examples/data_connectors). Please check them out!
