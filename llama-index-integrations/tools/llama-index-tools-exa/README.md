# LlamaIndex Tools Integration: Exa

This tool connects to [Exa](https://exa.ai/) to easily enable
your agent to search and get HTML content from the Internet.

To begin, you need to obtain an API key on the [Exa developer dashboard](https://dashboard.exa.ai).

## Usage

This tool has more a extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-exa/examples/exa.ipynb)

Here's an example usage of the ExaToolSpec.

```python
# %pip install llama-index llama-index-core llama-index-tools-exa

from llama_index.tools.exa import ExaToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

exa_tool = ExaToolSpec(
    api_key=os.environ["EXA_API_KEY"],
)
agent = FunctionAgent(
    tools=exa_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(
    await agent.run(
        "Can you summarize the news published in the last month on superconductors"
    )
)
```

`search`: Search for a list of articles relating to a natural language query

`retrieve_documents`: Retrieve a list of documents returned from `exa_search`.

`search_and_retrieve_documents`: Combines search and retrieve_documents to directly return a list of documents related to a search

`find_similar`: Find similar documents to a given URL.

`current_date`: Utility for the Agent to get todays date

This loader is designed to be used as a way to load data as a Tool in a Agent.
