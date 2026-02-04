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
from llama_index.tools.moss import MossToolSpec
from inferedge_moss import MossClient

# Initialize the client
# The client requires a Moss API key and a project ID
MOSS_API_KEY = os.getenv("MOSS_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
client = MossClient(project_id=PROJECT_ID, api_key=MOSS_API_KEY)

# Initialize the tool
# Note: You can customize top_k and alpha (hybrid search weight)
tool = MossToolSpec(client=client, index_name="my_index", top_k=5, alpha=0.5)

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
- `top_k` (int, default=5): Number of results to return.
- `alpha` (float, default=0.5): Weight for hybrid search (0.0=keyword, 1.0=semantic).
- `model_id` (str, default="moss-minilm"): The model ID to use for embeddings.

## Examples

The `examples/` directory contains:

- [Moss Agent Notebook](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/tools/llama-index-tools-moss/examples/moss_agent.ipynb): A Jupyter notebook walking through installation, indexing, and agent usage.
- [Moss Agent Script](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/tools/llama-index-tools-moss/examples/moss_agent.py): A concise Python script demonstrating the ReAct agent flow.
