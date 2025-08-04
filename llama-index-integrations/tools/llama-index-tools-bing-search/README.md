# Bing Search Tool

This tool connects to a Bing account and allows an Agent to perform searches for news, images and videos.

You will need to set up a search key using Azure,learn more here: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-bing-search/examples/bing_search.ipynb)

Here's an example usage of the BingSearchToolSpec.

```python
from llama_index.tools.bing_search import BingSearchToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = BingSearchToolSpec(api_key="your-key")

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(), llm=OpenAI(model="gpt-4.1")
)

print(await agent.run("what's the latest news about superconductors"))
print(await agent.run("what does lk-99 look like"))
print(await agent.run("is there any videos of it levitating"))
```

`bing_news_search`: Search for news results related to a query
`bing_image_search`: Search for images related to a query
`bing_video_search`: Search for videos related to a query

This loader is designed to be used as a way to load data as a Tool in a Agent.
