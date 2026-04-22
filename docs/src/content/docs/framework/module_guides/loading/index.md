---
title: Loading Data
---

The key to data ingestion in LlamaIndex is loading and transformations. Once you have loaded Documents, you can process them via transformations and output Nodes.

Once you have [learned about the basics of loading data](/python/framework/understanding/rag/loading) in our Understanding section, you can read on to learn more about:

### Loading

- [SimpleDirectoryReader](/python/framework/module_guides/loading/simpledirectoryreader), our built-in loader for loading all sorts of file types from a local directory
- [LlamaParse](/python/framework/module_guides/loading/connector/llama_parse), LlamaIndex's official tool for PDF parsing, available as a managed API.
- [LlamaHub](/python/framework/module_guides/loading/connector), our registry of hundreds of data loading libraries to ingest data from any source

### Transformations

This includes common operations like splitting text.

- [Node Parser Usage Pattern](/python/framework/module_guides/loading/node_parsers), showing you how to use our node parsers
- [Node Parser Modules](/python/framework/module_guides/loading/node_parsers/modules), showing our text splitters (sentence, token, HTML, JSON) and other parser modules.

### Putting it all Together

- [The ingestion pipeline](/python/framework/module_guides/loading/ingestion_pipeline) which allows you to set up a repeatable, cache-optimized process for loading data.

### Abstractions

- [Document and Node objects](/python/framework/module_guides/loading/documents_and_nodes) and how to customize them for more advanced use cases
