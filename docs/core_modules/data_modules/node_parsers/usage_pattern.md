# Usage Pattern

## Getting Started

Node parsers can be used on their own:

```python
from llama_index import Document
from llama_index.node_parser import SentenceAwareNodeParser

node_parser = SentenceAwareNodeParser(chunk_size=1024, chunk_overlap=20)

nodes = node_parser.get_nodes_from_documents([Document(text="long text")], show_progress=False)
```

Or set inside a `ServiceContext` to be used automatically when an index is constructed using `.from_documents()`:

```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.node_parser import SentenceAwareNodeParser

documents = SimpleDirectoryReader("./data").load_data()

node_parser = SentenceAwareNodeParser(chunk_size=1024, chunk_overlap=20)
service_context = ServiceContext.from_defaults(node_parser=node_parser)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)
```

## SentenceWindowNodeParser

The `SentenceWindowNodeParser` is similar to other node parsers, except that it splits all documents into individual sentences. The resulting nodes also contain the surrounding "window" of sentences around each node in the metadata. Note that this metadata will not be visible to the LLM or embedding model.

This is most useful for generating embeddings that have a very specific scope. Then, combined with a `MetadataReplacementNodePostProcessor`, you can replace the sentence with it's surrounding context before sending the node to the LLM.

An example of setting up the parser with default settings is below. In practice, you would usually only want to adjust the window size of sentences.

```python
import nltk
from llama_index.node_parser import SentenceWindowNodeParser

node_parser = SentenceWindowNodeParser.from_defaults(
  # how many sentences on either side to capture
  window_size=3,
  # the metadata key that holds the window of surrounding sentences
  window_metadata_key="window",
  # the metadata key that holds the original sentence
  original_text_metadata_key="original_sentence"
)
```

A full example can be found [here in combination with the `MetadataReplacementNodePostProcessor`](/examples/node_postprocessor/MetadataReplacementDemo.ipynb).
