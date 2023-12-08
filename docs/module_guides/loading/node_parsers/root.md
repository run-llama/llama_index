# Node Parser Usage Pattern

Node parsers are a simple abstraction that take a list of documents, and chunk them into `Node` objects, such that each node is a specific chunk of the parent document. When a document is broken into nodes, all of it's attributes are inherited to the children nodes (i.e. `metadata`, text and metadata templates, etc.). You can read more about `Node` and `Document` properties [here](/module_guides/loading/documents_and_nodes/root.md).

## Getting Started

### Standalone Usage

Node parsers can be used on their own:

```python
from llama_index import Document
from llama_index.node_parser import SentenceSplitter

node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

nodes = node_parser.get_nodes_from_documents(
    [Document(text="long text")], show_progress=False
)
```

### Transformation Usage

Node parsers can be included in any set of transformations with an ingestion pipeline.

```python
from llama_index import SimpleDirectoryReader
from llama_index.ingestion import IngestionPipeline
from llama_index.node_parser import TokenTextSplitter

documents = SimpleDirectoryReader("./data").load_data()

pipeline = IngestionPipeline(transformations=[TokenTextSplitter(), ...])

nodes = pipeline.run(documents=documents)
```

### Service Context Usage

Or set inside a `ServiceContext` to be used automatically when an index is constructed using `.from_documents()`:

```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.text_splitter import SentenceSplitter

documents = SimpleDirectoryReader("./data").load_data()

text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
service_context = ServiceContext.from_defaults(text_splitter=text_splitter)

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)
```

## Modules

```{toctree}
---
maxdepth: 2
---
modules.md
```
