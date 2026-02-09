# LlamaIndex Node Parser Chonkie Integration

This package provides an integration between [LlamaIndex](https://www.llamaindex.ai/) and [Chonkie](https://github.com/chonkie-inc/chonkie), a powerful and flexible chunking library.

## Installation

```bash
pip install llama-index-node_parser-chonkie
```

## Quick Start

```python
from llama_index.core import Document
from llama_index.node_parser.chonkie import Chunker

# Create a chunker (defaults to 'recursive')
chunker = Chunker(chunk_size=512)

# Create a document
doc = Document(text="Your long text here...")

# Get nodes
nodes = chunker.get_nodes_from_documents([doc])
```

## Supported Chunkers

The `Chunker` acts as a wrapper for various Chonkie chunking strategies. You can specify the strategy using the `chunker` parameter:

| `chunker`   | Description                                                           |
| ----------- | --------------------------------------------------------------------- |
| `recursive` | (Default) Recursively splits text based on a hierarchy of separators. |
| `sentence`  | Splits text into sentences.                                           |
| `token`     | Splits text into chunks based on token counts.                        |
| `word`      | Splits text based on word counts.                                     |
| `semantic`  | Splits text based on semantic similarity.                             |
| `late`      | Late chunking strategy.                                               |
| `neural`    | Neural-based chunking.                                                |
| `code`      | Optimized for source code.                                            |
| `fast`      | High-performance basic chunking.                                      |

run the following code to see the full list of valid aliases:

```python
from llama_index.node_parser import Chunker

print(Chunker.valid_chunkers)
```

## Advanced Configuration

You can pass any keyword arguments accepted by the underlying Chonkie chunker directly to `Chunker`:

```python
chunker = Chunker(
    chunker="semantic",
    chunk_size=512,
    embedding_model="all-MiniLM-L6-v2",
    threshold=0.5,
)
```

## Integration with Node Parsing

You can use `Chunker` directly to parse documents into nodes:

```python
from llama_index.core import Document
from llama_index.node_parser.chonkie import Chunker

chunker = Chunker(chunk_size=512)
doc = Document(text="Your long text here...")
nodes = chunker.get_nodes_from_documents([doc])
```

or you can also use it as a component within the Ingestion pipeline:

```python
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.node_parser.chonkie import Chunker

pipeline = IngestionPipeline(
    transformations=[
        Chunker("recursive", chunk_size=512),
        # ... other transformations
    ]
)

nodes = pipeline.run(documents=[Document.example()])
```
