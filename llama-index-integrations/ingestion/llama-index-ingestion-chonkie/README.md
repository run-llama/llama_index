# LlamaIndex Ingestion Chonkie Integration

This package provides an integration between [LlamaIndex](https://www.llamaindex.ai/) and [Chonkie](https://github.com/chonkie-inc/chonkie), a powerful and flexible chunking library.

## Installation

```bash
pip install llama-index-ingestion-chonkie
```

## Quick Start

```python
from llama_index.core import Document
from llama_index.ingestion.chonkie import ChonkieChunker

# Create a chunker (defaults to 'recursive')
chunker = ChonkieChunker(chunk_size=512, chunk_overlap=50)

# Create a document
doc = Document(text="Your long text here...")

# Get nodes
nodes = chunker.get_nodes_from_documents([doc])
```

## Supported Chunkers

The `ChonkieChunker` acts as a wrapper for various Chonkie chunking strategies. You can specify the strategy using the `chunker_type` parameter:

| `chunker_type` | Description |
|----------------|-------------|
| `recursive` | (Default) Recursively splits text based on a hierarchy of separators. |
| `sentence` | Splits text into sentences. |
| `token` | Splits text into chunks based on token counts. |
| `word` | Splits text based on word counts. |
| `semantic` | Splits text based on semantic similarity. |
| `late` | Late chunking strategy. |
| `neural` | Neural-based chunking. |
| `code` | Optimized for source code. |
| `fast` | High-performance basic chunking. |

## Advanced Configuration

You can pass any keyword arguments accepted by the underlying Chonkie chunker directly to `ChonkieChunker`:

```python
chunker = ChonkieChunker(
    chunker_type="semantic",
    chunk_size=512,
    embedding_model="all-MiniLM-L6-v2",
    threshold=0.5
)
```

## Integration with IngestionPipeline
<!-- see https://developers.llamaindex.ai/python/framework/module_guides/loading/ingestion_pipeline/#usage-pattern for reference -->
```python
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.ingestion.chonkie import ChonkieChunker

pipeline = IngestionPipeline(
    transformations=[
        ChonkieChunker(chunker_type="recursive", chunk_size=512),
        # ... other transformations
    ]
)

nodes = pipeline.run(documents=[Document.example()])
```
