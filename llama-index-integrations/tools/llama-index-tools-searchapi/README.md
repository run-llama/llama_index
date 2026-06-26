# LlamaIndex Tools Integration: SearchApi

This tool enables agents to search the web in real time using
[SearchApi](https://www.searchapi.io), a SERP API that returns structured results
from 100+ engines, including Google, Google News, Google Scholar, Bing, Baidu, and YouTube.

You will need to set up a SearchApi account to get an API key. You can get one here:
https://www.searchapi.io

## Usage

Here's an example usage of the `SearchApiToolSpec`.

```python
from llama_index.tools.searchapi import SearchApiToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = SearchApiToolSpec(api_key="your-key")

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(await agent.run("what's the latest news about superconductors"))
```

You can also choose a specific engine when creating the tool, or per query:

```python
# Default every query to Google News
tool_spec = SearchApiToolSpec(api_key="your-key", engine="google_news")

# Or override the engine for a single search
results = tool_spec.search(
    "lk-99 levitating", engine="youtube", num_results=10
)
```

`search`: Search a given engine and return a list of results as documents.

This loader is designed to be used as a way to load data as a Tool in an Agent.
