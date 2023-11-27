# Loading

The key to data ingestion in LlamaIndex is the Loaders. Once you have data, you may further refine your Documents and Nodes.

Once you have [learned about the basics of loading data](/understanding/loading/loading.html) in our Understanding section, you can read on to learn more about:

- [SimpleDirectoryReader](simpledirectoryreader.md), our built-in loader for loading all sorts of file types from a local directory
- [LlamaHub](connector/root.md), our registry of hundreds of data loading libraries to ingest data from any source
- [Document and Node objects](documents_and_nodes/root.md) and how to customize them for more advanced use cases
- [Node parsers](node_parsers/root.md), our set of helper classes to generate nodes from raw text and files
- [The ingestion pipeline](ingestion_pipeline/root.md) which allows you to set up a repeatable, cache-optimized process for loading data.

```{toctree}
---
maxdepth: 1
hidden: true
---
connector/root.md
documents_and_nodes/root.md
node_parsers/root.md
ingestion_pipeline/root.md
```
