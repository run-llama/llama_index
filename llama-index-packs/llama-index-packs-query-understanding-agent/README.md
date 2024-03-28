# LlamaIndex Packs Integration: Query Understanding Agent

This LlamaPack implements Query Understanding Agent

Taking inspiration from Humans - when asked a query, humans would clarify what the query means before proceeding if the human sensed the query is unclear. This LlamaPack implements this.

Check out the [full notebook here](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-query-understanding-agent/examples/query_understanding_agent.ipynb).

### Installation

```bash
pip install llama-index
```

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack QueryUnderstandingAgent --download-dir ./query_understanding_agent
```

You can then inspect the files at `./query_understanding_agent` and use them as a template for your own project.

## Code Usage

You can download the pack to a the `./query_understanding_agent` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
QueryUnderstandingAgentPack = download_llama_pack(
    "QueryUnderstandingAgent", "./query_understanding_agent"
)

# You can use any llama-hub loader to get documents!
```

From here, you can use the pack, or inspect and modify the pack in `./query_understanding_agent`.
See example notebook for usage.
