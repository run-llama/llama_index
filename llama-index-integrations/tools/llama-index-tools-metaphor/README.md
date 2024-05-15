# Metaphor Tool

# ================================================================================

NOTE: Deprecated in favor of Exa Tool.

# ================================================================================

This tool connects to [Metaphor](https://metaphor.systems/) to easily enable
your agent to search and get HTML content from the Internet.

To begin, you need to obtain an API key on the [Metaphor developer dashboard](https://dashboard.metaphor.systems).

## Usage

This tool has more a extensive example usage documented in a Jupyter notebook [here](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/tools/notebooks/metaphor.ipynb)

Here's an example usage of the MetaphorToolSpec.

```python
from llama_index.tools.metaphor import MetaphorToolSpec
from llama_index.agent.openai import OpenAIAgent

metaphor_tool = MetaphorToolSpec(
    api_key="your-key",
)
agent = OpenAIAgent.from_tools(metaphor_tool.to_tool_list())

agent.chat(
    "Can you summarize the news published in the last month on superconductors"
)
```

`search`: Search for a list of articles relating to a natural language query
`retrieve_documents`: Retrieve a list of documents returned from `metaphor_search`.
`search_and_retrieve_documents`: Combines search and retrieve_documents to directly return a list of documents related to a search
`find_similar`: Find similar documents to a given URL.
`current_date`: Utility for the Agent to get todays date

This loader is designed to be used as a way to load data as a Tool in a Agent.
