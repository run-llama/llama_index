# LoadAndSearch Tool

```bash
pip install llama-index-tools-wikipedia
```

This Tool Spec is intended to wrap other tools, allowing the Agent to perform separate loading and reading of data. This is very useful for when tools return information larger than or closer to the size of the context window.

## Usage

Here's an example usage of the LoadAndSearchToolSpec.

```python
from llama_index.core.tools.tool_spec.load_and_search import (
    LoadAndSearchToolSpec,
)
from llama_index.core.agent import OpenAIAgent
from llama_index.tools.wikipedia.base import WikipediaToolSpec

wiki_spec = WikipediaToolSpec()

# Get the search_data tool from the wikipedia tool spec
tool = wiki_spec.to_tool_list()[1]

# Wrap the tool, splitting into a loader and a reader
agent = OpenAIAgent.from_tools(
    LoadAndSearchToolSpec.from_defaults(tool).to_tool_list(), verbose=True
)

agent.chat("who is ben affleck married to")
```

`load`: Calls the wrapped function and loads the data into an index
`read`: Searches the index for the specified query

This loader is designed to be used as a way to load data as a Tool in a Agent.
