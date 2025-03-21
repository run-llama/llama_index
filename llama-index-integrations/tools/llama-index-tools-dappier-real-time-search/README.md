# Dappier Real Time Search Tool

[Dappier](https://dappier.com) is a platform that connects LLMs and Agentic AI agents to real-time, rights-cleared data from trusted sources, including web search, finance, and news. By providing enriched, prompt-ready data, Dappier empowers AI with verified and up-to-date information for a wide range of applications. Explore a wide range of data models in our marketplace at [marketplace.dappier.com](https://marketplace.dappier.com).

## Overview

The **Dappier Real Time Search Tool** is a powerful API-driven tool designed to fetch up-to-the-minute information from various sources, including real-time web search results and stock market data. It leverages Dappier's AI capabilities to deliver timely insights, making it an essential asset for businesses, financial analysts, and users seeking real-time data.

## Key Features

- **Real-Time Web Search**: Fetches the latest news, weather updates, travel deals, and other relevant web content through AI-powered search.
- **Stock Market Insights**: Retrieves real-time stock prices, financial news, and trade data from **Polygon.io**, enriched with AI-driven insights.
- **AI-Powered Queries**: Uses pre-defined AI models to refine and enhance search results for better accuracy and relevance.
- **Seamless Integration**: Works with the Dappier API, requiring a valid API key to access search functionalities.

## Installation

pip install llama-index-tools-dappier-real-time-search

## Setup

You'll need to set up your API keys for OpenAI and Dappier.

Your can go to [here](https://platform.openai.com/settings/organization/api-keys) to get API Key from Open AI.

```python
os.environ["OPENAI_API_KEY"] = "openai_api_key"
```

You can go to [here](https://platform.dappier.com/profile/api-keys) to get API Key from Dappier with **free** credits.

```python
os.environ["DAPPIER_API_KEY"] = "dappier_api_key"
```

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-dappier-real-time-search/examples/dappier_real_time_search.ipynb)

Here's an example usage of the DappierRealTimeSearchToolSpec.

```python
from llama_index.tools.dappier_real_time_search import (
    DappierRealTimeSearchToolSpec,
)
from llama_index.core.agent import FunctionCallingAgent
from llama_index.llms.openai import OpenAI

dappier_tool = DappierRealTimeSearchToolSpec()
agent = FunctionCallingAgent.from_tools(
    dappier_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
)

agent.chat(
    "How is the weather in Boston today ? Create a detailed analysis in markdown format."
)
```

The tools available in are:

- `search_real_time_data`: A tool that performs a real-time web search to retrieve the latest information, including news, weather, travel deals, and more.

- `search_stock_market_data`: A tool that fetches real-time stock market data, including stock prices, financial news, and trade updates, with AI-powered insights.

This loader is designed to be used as a way to load data as a Tool in an Agent.
