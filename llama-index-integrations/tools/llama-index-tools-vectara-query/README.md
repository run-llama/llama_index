## Vectara Query Tool

This tool connects to a Vectara corpus and allows agents to make semantic search or retrieval augmented generation (RAG) queries.

## Usage

This tool has a more extensive example usage documented in a Jupyter notebok [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-vectara-query/examples/vectara_query.ipynb)

To use this tool, you'll need the following information in your environment:

- `VECTARA_CUSTOMER_ID`: The customer id for your Vectara account. If you don't have an account, you can create one [here](https://vectara.com/integrations/llamaindex).
- `VECTARA_CORPUS_ID`: The corpus id for the Vectara corpus that you want your tool to search for information. If you need help creating a corpus with your data, follow this [Quick Start](https://docs.vectara.com/docs/quickstart) guide.
- `VECTARA_API_KEY`: An API key that can perform queries on this corpus.

Here's an example usage of the VectaraQueryToolSpec.

```python
from llama_index.tools.vectara_query import VectaraQueryToolSpec
from llama_index.agent.openai import OpenAIAgent

# Connecting to a Vectara corpus about Electric Vehicles
tool_spec = VectaraQueryToolSpec()

agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat("What are the different types of electric vehicles?")
```

The available tools are:

`semantic_search`: A tool that accepts a query and uses semantic search to obtain the top search results.

`rag_query`: A tool that accepts a query and uses RAG to obtain a generative response grounded in the search results.
