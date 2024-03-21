# DuckDuckGo Search Tool

This tool enables agents to search and retrieve results from the DuckDuckGo search engine. It utilizes the duckduckgo_search package, which either fetches instant query answers from DuckDuckGo or conducts a full search and parses the results.

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](./examples/duckduckgo_search.ipynb)

Here's an example usage of the DuckDuckGoSearchToolSpec.

```python
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.agent.openai import OpenAIAgent

tool_spec = DuckDuckGoSearchToolSpec()

agent = OpenAIAgent.from_tools(DuckDuckGoSearchToolSpec.to_tool_list())

agent.chat("What's going on with the superconductor lk-99")
agent.chat("what are the latest developments in machine learning")
```

## Available tool functions:

- `duckduckgo_instant_search`: Make a query to DuckDuckGo api to receive an instant answer.

- `duckduckgo_full_search`: Make a query to DuckDuckGo search to receive a full search results.
