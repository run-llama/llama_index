# Node Parser

## Concept

Node parsers are a simple abstraction that take a list of documents, and chunk them into `Node` objects, such that each node is a specific size. When a document is broken into nodes, all of it's attributes are inherited to the children nodes (i.e. `metadata`, text and metadata templates, etc.). You can read more about `Node` and `Document` properies [here](/core_modules/data_modules/documents_and_nodes/root.md).

A node parser can configure the chunk size (in tokens) as well as any overlap between chunked nodes. The chunking is done by using a `TokenTextSplitter`, which default to a chunk size of 1024 and a default chunk overlap of 20 tokens.

# Usage Pattern

```python
from llama_index.node_parser import SimpleNodeParser

node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)
```

You can find more usage details and availbale customization options below.

```{toctree}
---
maxdepth: 1
---
usage_pattern.md
```
