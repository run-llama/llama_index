# Bing Search Tool

This tool connects to a Bing account and allows an Agent to perform searches for news, images and videos.

You will need to set up a search key using Azure,learn more here: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/tools/notebooks/bing_search.ipynb)

Here's an example usage of the BingSearchToolSpec.

```python
from llama_index.tools.bing_search import BingSearchToolSpec
from llama_index.agent.openai import OpenAIAgent

tool_spec = BingSearchToolSpec(api_key="your-key")

agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat("what's the latest news about superconductors")
agent.chat("what does lk-99 look like")
agent.chat("is there any videos of it levitating")
```

`bing_news_search`: Search for news results related to a query
`bing_image_search`: Search for images related to a query
`bing_video_search`: Search for videos related to a query

This loader is designed to be used as a way to load data as a Tool in a Agent.
