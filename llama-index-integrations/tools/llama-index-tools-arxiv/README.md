# ArXiv Search Tool

This tool connects to ArXiv and allows an Agent to search for recent papers and their summaries to retrieve recent information on mathematical and scientific information

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-arxiv/examples/arxiv.ipynb).

Here's an example usage of the ArxivToolSpec.

```python
from llama_index.tools.arxiv import ArxivToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = ArxivToolSpec()

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(), llm=OpenAI(model="gpt-4.1")
)

await agent.run("What's going on with the superconductor lk-99")
await agent.run("what are the latest developments in machine learning")
```

`arxiv_query`: Search arXiv for results related to the query

This loader is designed to be used as a way to load data as a Tool in a Agent.
