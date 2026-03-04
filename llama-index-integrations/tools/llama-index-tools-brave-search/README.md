# LlamaIndex Tools Integration: Brave_Search

This tool enables agents to search and retrieve results from the Brave search engine.

You will need to set up an Brave account to get an search api key. Please check more here: https://brave.com/search/api

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](./examples/brave_search.ipynb)

Here's an example usage of the BraveSearchToolSpec.

```python
from llama_index.tools.brave_search import BraveSearchToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = BraveSearchToolSpec(api_key="your-key")

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

answer = await agent.run("what's the latest news about superconductors")
answer = await agent.run("what does lk-99 look like")
answer = await agent.run("is there any videos of it levitating")
```
