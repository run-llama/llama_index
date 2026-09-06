# LlamaIndex Tools Integration: Desearch

This tool connects to Desearch to enable your agent to perform searches across various platforms like web, Twitter, and more.

To begin, you need to obtain an API key from the Desearch developer dashboard.

Website: https://desearch.io

## Usage

Here's an example usage of the `DesearchToolSpec`.

To get started, you will need a [Desearch API key](https://console.desearch.ai/api-keys)

```python
# %pip install llama-index llama-index-core llama-index-tools-desearch

import os

from llama_index.tools.desearch import DesearchToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

desearch_tool = DesearchToolSpec(
    api_key=os.environ["DESEARCH_API_KEY"],
)
agent = FunctionAgent(
    tools=desearch_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(await agent.run("Can you find the latest news on quantum computing?"))
```

### Available Functions

- **ai_search_tool**: Perform a search using Desearch across multiple platforms like web, Twitter, Reddit, etc.
- **twitter_search_tool**: Perform a basic Twitter search using the Desearch API.
- **web_search_tool**: Perform a basic web search using the Desearch API.

This loader is designed to be used as a way to load data as a Tool in an Agent.
