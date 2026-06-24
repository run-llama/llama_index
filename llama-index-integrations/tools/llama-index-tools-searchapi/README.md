# LlamaIndex Tools Integration: SearchApi

This tool enables agents to run real-time searches through [SearchApi](https://www.searchapi.io),
a SERP API with access to 100+ engines including Google, Google News, Google Scholar,
YouTube, and Google Jobs.

You will need a SearchApi API key. Get one at https://www.searchapi.io.

## Usage

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

You can target a specific engine, either as the tool default or per query:

```python
# Set the default engine for every search this tool runs.
news = SearchApiToolSpec(api_key="your-key", engine="google_news")

# Or override the engine for a single call.
tool_spec = SearchApiToolSpec(api_key="your-key")
tool_spec.search("retrieval augmented generation", engine="google_scholar")
```
