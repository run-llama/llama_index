# LlamaIndex Node_Parser-Relational Integration: Dashscope

Transform your documents into nodes with ease using the Dashscope integration for LlamaIndex. This tool allows for precise control over chunk size, overlap size, and more, tailored for the Dashscope reader output format.

## Installation

```shell
pip install llama-index-node-parser-dashscope
```

## Quick Start

Get up and running with just a few lines of code:

```python
import json
import os
from llama_index.node_parser.relational.dashscope import (
    DashScopeJsonNodeParser,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import Document

# Set your Dashscope API key in the environment
os.environ["DASHSCOPE_API_KEY"] = "your_api_key_here"

documents = [
    # Prepare your documents obtained from the Dashscope reader
]

# Initialize the DashScope JsonNodeParser
node_parser = DashScopeJsonNodeParser(
    chunk_size=100, overlap_size=0, separator=" |,|，|。|？|！|\n|\?|\!"
)

# Set up the ingestion pipeline with the node parser
pipeline = IngestionPipeline(transformations=[node_parser])

# Process the documents and print the resulting nodes
nodes = pipeline.run(documents=documents, show_progress=True)
for node in nodes:
    print(node)
```

## Configuration

- API Key: You need a Dashscope API key to begin. Set it in your environment as shown in the Quick Start section.
- Document Preparation: Your documents must be in the Dashscope reader output format.
