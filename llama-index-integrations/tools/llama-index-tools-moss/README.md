# Moss Tool

This tool provides integration with Moss, a real-time semantic search engine.

## Installation

```bash
pip install llama-index-tools-moss
```

## Usage

You can use the `MossToolSpec` to interact with your Moss index.

### Initialization

```python
import os
from llama_index.tools.moss import MossToolSpec, QueryOptions
from inferedge_moss import MossClient

# Initialize the client
# The client requires a Moss project key and a project ID
MOSS_PROJECT_KEY = os.getenv("MOSS_PROJECT_KEY")
MOSS_PROJECT_ID = os.getenv("MOSS_PROJECT_ID")
client = MossClient(project_id=MOSS_PROJECT_ID, project_key=MOSS_PROJECT_KEY)

# Initialize the tool
# Note: You can customize top_k and alpha through query_options (hybrid search weight)
options = QueryOptions(alpha=0.6, top_k=9, model_id="moss-minilm")
tool = MossToolSpec(
    client=client, index_name="my_index", query_options=options
)

# Convert to tool list for agents
tools = tool.to_tool_list()
```

### Indexing Documents

You can index documents into your Moss index using the `index_docs` method:

```python
from inferedge_moss import DocumentInfo

docs = [
    DocumentInfo(
        text="LlamaIndex is great!", metadata={"source": "review.txt"}
    ),
    DocumentInfo(text="Moss is fast!", metadata={"source": "specs.txt"}),
]

# Index the documents
await tool.index_docs(docs)
```

### Parameters

- `client` (MossClient): The initialized Moss client.
- `index_name` (str): The name of the index to query.
- `query_options` (QueryOptions): Configuration options for the tool (optional).
  - `top_k` (int, default=5): Number of results to return.
  - `alpha` (float, default=0.5): Weight for hybrid search (0.0=keyword, 1.0=semantic).
  - `model_id` (str, default="moss-minilm"): The model ID to use for embeddings.

### MODEL IDs

- `moss-minilm`: Fast, lightweight (default). Best for speed-first, edge/offline use.
- `moss-mediumlm`: Higher accuracy with reasonable performance. Best when search quality is important.

## Examples

The `examples/` directory contains:

- [Moss Agent Notebook](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/tools/llama-index-tools-moss/examples/moss_agent.ipynb): A Jupyter notebook walking through installation, indexing, and agent usage.
- [Moss Agent Script](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/tools/llama-index-tools-moss/examples/moss_agent.py): A concise Python script demonstrating the ReAct agent flow.
