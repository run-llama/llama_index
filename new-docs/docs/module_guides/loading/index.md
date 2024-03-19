# Loading Data

The key to data ingestion in LlamaIndex is loading and transformations. Once you have loaded Documents, you can process them via transformations and output Nodes.

Once you have [learned about the basics of loading data](../../understanding/loading/loading.md) in our Understanding section, you can read on to learn more about:

### Loading

- [SimpleDirectoryReader](simpledirectoryreader.md), our built-in loader for loading all sorts of file types from a local directory
- [LlamaParse](connector/llama_parse.md), LlamaIndex's official tool for PDF parsing, available as a managed API.
- [LlamaHub](connector/index.md), our registry of hundreds of data loading libraries to ingest data from any source

### Transformations

This includes common operations like splitting text.

- [Node Parser Usage Pattern](node_parsers/index.md), showing you how to use our node parsers
- [Node Parser Modules](node_parsers/modules.md), showing our text splitters (sentence, token, HTML, JSON) and other parser modules.

### Putting it all Together

- [The ingestion pipeline](ingestion_pipeline/index.md) which allows you to set up a repeatable, cache-optimized process for loading data.

### Abstractions

- [Document and Node objects](documents_and_nodes/index.md) and how to customize them for more advanced use cases
