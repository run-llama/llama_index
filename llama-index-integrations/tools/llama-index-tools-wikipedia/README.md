# Wikipedia Tool

This tool fetches content from wikipedia and makes it available to the agent as a Tool. You can search for pages or load pages directly.

## Usage

This tool has more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-wikipedia/examples/wikipedia.ipynb)

```python
from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = WikipediaToolSpec()

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(), llm=OpenAI(model="gpt-4.1")
)

print(await agent.run("Who is Ben Afflecks spouse?"))
```

`load_data`: Loads a page from wikipedia
`search_data`: Searches wikipedia for a query and loads all matching pages

This loader is designed to be used as a way to load data as a Tool in a Agent.
See [this LlamaIndex tutorial][1] for examples.

[1]: https://gpt-index.readthedocs.io/en/stable/examples/index_structs/knowledge_graph/KnowledgeGraphIndex_vs_VectorStoreIndex_vs_CustomIndex_combined.html#load-data-from-wikipedia
