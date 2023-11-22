# Transformations

A transformation is something that takes a list of nodes as an input, and returns a list of nodes. Each component that implements the `Transformation` base class has both a synchronous `__call__()` definition and an async `acall()` definition.

Currently, the following components are `Transformation` objects:

- [`TextSplitter`](text_splitters)
- [`NodeParser`](/module_guides/loading/node_parsers/modules.md)
- [`MetadataExtractor`](/module_guides/loading/documents_and_nodes/usage_metadata_extractor.md)
- `Embeddings`model (check our [list of supported embeddings](list_of_embeddings))

## Usage Pattern

While transformations are best used with with an [`IngestionPipeline`](./root.md), they can also be used directly.

```python
from llama_index.text_splitter import SentenceSplitter
from llama_index.extractors import TitleExtractor

node_parser = SentenceSplitter(chunk_size=512)
extractor = TitleExtractor()

# use transforms directly
nodes = node_parser(documents)

# or use a transformation in async
nodes = await extractor.acall(nodes)
```

## Combining with ServiceContext

Transformations can be passed into a service context, and will be used when calling `from_documents()` or `insert()` on an index.

```python
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.ingestion import IngestionPipeline
from llama_index.text_splitter import TokenTextSplitter

transformations = [
    TokenTextSplitter(chunk_size=512, chunk_overlap=128),
    TitleExtractor(nodes=5),
    QuestionsAnsweredExtractor(questions=3),
]

service_context = ServiceContext.from_defaults(
    transformations=[text_splitter, title_extractor, qa_extractor]
)

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)
```

## Custom Transformations

You can implement any transformation yourself by implementing the base class.

The following custom transformation will remove any special characters or punctutaion in text.

```python
import re
from llama_index import Document
from llama_index.embeddings import OpenAIEmbedding
from llama_index.text_splitter import SentenceSplitter
from llama_index.ingestion import IngestionPipeline
from llama_index.schema import TransformComponent


class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r"[^0-9A-Za-z ]", "", node.text)
        return nodes
```

These can then be used directly or in any `IngestionPipeline`.

```python
# use in a pipeline
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TextCleaner(),
        OpenAIEmbedding(),
    ],
)

nodes = pipeline.run(documents=[Document.example()])
```
