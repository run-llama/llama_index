# Transformations

A transformation is something that takes a list of nodes as an input, and returns a list of nodes. Each component that implements the `Transformation` base class has both a synchronous `__call__()` definition and an async `acall()` definition.

Currently, the following components are `Transformation` objects:

- [`TextSplitter`](/python/framework/module_guides/loading/node_parsers/modules#text-splitters)
- [`NodeParser`](/python/framework/module_guides/loading/node_parsers/modules)
- [`MetadataExtractor`](/python/framework/module_guides/loading/documents_and_nodes/usage_metadata_extractor)
- `Embeddings`model (check our [list of supported embeddings](/python/framework/module_guides/models/embeddings#list-of-supported-embeddings))

## Usage Pattern

While transformations are best used with with an [`IngestionPipeline`](/python/framework/module_guides/loading/ingestion_pipeline), they can also be used directly.

```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor

node_parser = SentenceSplitter(chunk_size=512)
extractor = TitleExtractor()

# use transforms directly
nodes = node_parser(documents)

# or use a transformation in async
nodes = await extractor.acall(nodes)
```

## Combining with An Index

Transformations can be passed into an index or overall global settings, and will be used when calling `from_documents()` or `insert()` on an index.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter

transformations = [
    TokenTextSplitter(chunk_size=512, chunk_overlap=128),
    TitleExtractor(nodes=5),
    QuestionsAnsweredExtractor(questions=3),
]

# global
from llama_index.core import Settings

Settings.transformations = [text_splitter, title_extractor, qa_extractor]

# per-index
index = VectorStoreIndex.from_documents(
    documents, transformations=transformations
)
```

## Custom Transformations

You can implement any transformation yourself by implementing the base class.

The following custom transformation will remove any special characters or punctuation in text.

```python
import re
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TransformComponent


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
