# Agent-Search Retrieval Pack

This LlamaPack creates a custom retriever that uses the agent-search API for retrieving general content indexed from the internet.

This framework facilitates seamless integration with the AgentSearch dataset (terabytes of indexed data!) or hosted search APIs (e.g. Search Engines).

During query-time, the user passes in the query string, search provider (`bing`, `agent-search`), and relevant nodes are retrieved from the hosted dataset.

To learn more, please refer to the documentation [here](https://agent-search.readthedocs.io/en/latest/).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack AgentSearchRetrieverPack --download-dir ./agent_search_pack
```

You can then inspect the files at `./agent_search_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a the `./agent_search_pack` directory:

```python
# Optionally set the API key in the env
# import os
# os.environ["SCIPHI_API_KEY"] = "..."

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
AgentSearchRetrieverPack = download_llama_pack(
    "AgentSearchRetrieverPack", "./agent_search_pack"
)

agent_search_pack = AgentSearchRetrieverPack(
    api_key="...", similarity_top_k=4, search_provider="agent-search"
)

# use the retriever directly
retriever = agent_search_pack.retriever
source_nodes = retriever.retrieve("query str")

# uses the agent-search retriever within a llama-index query engine!
query_engine = RetrieverQueryEngine.from_args(retriever)
response = query_engine.query("query str")
```

The `run()` function is a light wrapper around `retriever.retrieve()`.

```python
source_nodes = agent_search_pack.run("What can you tell me about LLMs?")

print(source_nodes)
```

See the [notebook on llama-hub](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/dense_x_retrieval/dense_x_retrieval.ipynb) for a full example.
