# Usage Pattern

## Getting Started

Node parsers can be used on their own:

```python
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser

node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)

nodes = node_parser.get_nodes_from_documents([Document(text="long text")], show_progress=False)
```

Or set inside a `ServiceContext` to be used automatically when an index is constructed using `.from_documents()`:

```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.node_parser import SimpleNodeParser

documents = SimpleDirectoryReader("./data").load_data()

node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)
service_context = ServiceContext.from_defaults(node_parser=node_parser)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)
```

## Customization

There are several options available to customize:

- `text_splitter` (defaults to `TokenTextSplitter`) - the text splitter used to split text into chunks.
- `include_metadata` (defaults to `True`) - whether or not `Node`s should inherit the document metadata.
- `include_prev_next_rel` (defaults to `True`) - whether or not to include previous/next relationships between chunked `Node`s
- `metadata_extractor` (defaults to `None`) - extra processing to extract helpful metadata. See [here for details](/core_modules/data_modules/documents_and_nodes/usage_metadata_extractor.md).

If you don't want to change the `text_splitter`, you can use `SimpleNodeParser.from_defaults()` to easily change the chunk size and chunk overlap. The defaults are 1024 and 20 respectively.

```python
from llama_index.node_parser import SimpleNodeParser

node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)
```

### Text Splitter Customization

If you do customize the `text_splitter` from the default `TokenTextSplitter`, you can use any splitter from langchain, or optionally our `SentenceSplitter`. Each text splitter has options for the default separator, as well as options for backup separators. These are useful for languages that are sufficiently different from English.

`TokenTextSplitter` configuration:

```python
from llama_index.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(
  separator=" ",
  chunk_size=1024,
  chunk_overlap=20,
  backup_separators=["\n"]
)

node_parser = SimpleNodeParser(text_splitter=text_splitter)
```

`SentenceSplitter` configuration:

```python
from llama_index.text_splitter import SentenceSplitter

text_splitter = SentenceSplitter(
  separator=" ",
  chunk_size=1024,
  chunk_overlap=20,
  backup_separators=["\n"],
  paragraph_separator="\n\n\n"
)

node_parser = SimpleNodeParser(text_splitter=text_splitter)
```

## SentenceWindowNodeParser

The `SentenceWindowNodeParser` is similar to the `SimpleNodeParser`, except that it splits all documents into individual sentences. Then, node also contains the surrounding "window" of setnences around it in the metadata. Note that this metadata will not be visible to the LLM or embedding model.

This is most useful for generating embeddings that have a very specific scope. Then, combined with a `MetadataReplacementNodePostProcessor`, you can replace the sentence with it's surrounding context before sending the node to the LLM. 

An example of setting up the parser with default settings is below. In practice, you would usually only want to adjust the

```python
import nltk
from llama_index.node_parser import SentenceWindowNodeParser

node_parser = SentenceWindowNodeParser.from_defaults(
  # how many sentences on either side to capture
  window_size=5,  
  # the metadata key that holds the window of surrounding sentences
  window_metadata_key="window",  
  # the metadata key that holds the original sentence
  original_text_metadata_key="original_sentence"
)
```

A full example can be found [here in combination with the `MetadataReplacementNodePostProcessor`](/examples/node_postprocessor/MetadataReplacementDemo.ipynb).
