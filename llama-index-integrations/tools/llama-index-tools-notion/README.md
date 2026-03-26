# Notion Tool

This tool loads and updates documents from Notion. The user specifies an API token to initialize the NotionToolSpec.

## Usage

This tool has more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-notion/examples/notion.ipynb)

Here's an example usage of the NotionToolSpec.

```python
from llama_index.tools.notion import NotionToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = NotionToolSpec()

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(await agent.run("Append the heading 'I am legend' to the movies page"))
```

`load_data`: Loads a list of page or databases by id
`search_data`: Searches for matching pages or databases
`append_data`: Appends content to the matching page or database

This loader is designed to be used as a way to load data as a Tool in a Agent.
