# LlamaIndex Tools Integration: Brave_Search

This tool enables agents to search and retrieve results from the Brave search engine.

You will need to set up an Brave account to get an search api key. Please check more here: https://brave.com/search/api

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](./examples/brave_search.ipynb)

Here's an example usage of the BraveSearchToolSpec.

```python
from llama_index.tools.brave_search import BraveSearchToolSpec
from llama_index.agent.openai import OpenAIAgent

tool_spec = BraveSearchToolSpec(api_key="your-key")

agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat("what's the latest news about superconductors")
agent.chat("what does lk-99 look like")
agent.chat("is there any videos of it levitating")
```
