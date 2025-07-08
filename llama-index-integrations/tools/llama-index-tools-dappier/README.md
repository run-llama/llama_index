# Dappier Tools

[Dappier](https://dappier.com) is a platform that connects LLMs and Agentic AI agents to real-time, rights-cleared data from trusted sources, including web search, finance, and news. By providing enriched, prompt-ready data, Dappier empowers AI with verified and up-to-date information for a wide range of applications. Explore a wide range of data models in our marketplace at [marketplace.dappier.com](https://marketplace.dappier.com).

## Overview

This package provides two tool specs:

- **DappierRealTimeSearchToolSpec**: A powerful API-driven tool designed to fetch up-to-the-minute information from various sources, including real-time web search results, stock market data, news, weather, travel deals and much more.

- **DappierAIRecommendationsToolSpec**: Provides AI-powered content recommendations across a range of domains including sports news, lifestyle news, pet care content from iHeartDogs and iHeartCats, compassionate living from GreenMonster, and local news from WISH-TV and 9&10 News.

## Key Features

### Real Time Search Tool

- **Real-Time Web Search**: Fetches the latest news, weather updates, travel deals, and other relevant web content through AI-powered search.

- **Stock Market Insights**: Retrieves real-time stock prices, financial news, and trade data from **Polygon.io**, enriched with AI-driven insights.

- **AI-Powered Queries**: Uses pre-defined AI models to refine and enhance search results for better accuracy and relevance.

- **Seamless Integration**: Works with the Dappier API, requiring a valid API key to access search functionalities.

### AI Recommendations Tool

- **Domain-specific recommendations**: Tailors AI-powered content suggestions across verticals like sports, lifestyle, pet care and news.

- **Smart search algorithms**: Supports modes like semantic, trending, and most recent to deliver the most relevant and timely results.

- **Reference domain targeting**: Lets you prioritize results from a specific site or domain for more context-aware recommendations.

- **Readable, structured output**: Returns responses with clear formatting, including title, summary, author, publish date, source, and links.

## Installation

```bash
pip install llama-index-tools-dappier
```

## Setup

You'll need to set up your API keys for OpenAI and Dappier.

You can go to [here](https://platform.openai.com/settings/organization/api-keys) to get API Key from Open AI.

```python
os.environ["OPENAI_API_KEY"] = "openai_api_key"
```

You can go to [here](https://platform.dappier.com/profile/api-keys) to get API Key from Dappier with **free** credits.

```python
os.environ["DAPPIER_API_KEY"] = "dappier_api_key"
```

## Usage

### Real Time Search Tool

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-dappier/examples/dappier_real_time_search.ipynb)

Here's an example usage of the DappierRealTimeSearchToolSpec.

```python
from llama_index.tools.dappier import (
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

### AI Recommendations Tool

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-dappier/examples/dappier_ai_recommendations.ipynb)

Here's an example usage of the DappierAIRecommendationsToolSpec.

```python
from llama_index.tools.dappier import (
    DappierAIRecommendationsToolSpec,
)
from llama_index.core.agent import FunctionCallingAgent
from llama_index.llms.openai import OpenAI

dappier_tool = DappierAIRecommendationsToolSpec()
agent = FunctionCallingAgent.from_tools(
    dappier_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
)

agent.chat(
    "Get latest sports news, lifestyle news, breaking news, dog care advice and summarize it into different sections, with source links."
)
```

The tools available in are:

- `get_sports_news_recommendations`: A tool that fetches real-time news, updates, and personalized content from top sports sources like Sportsnaut, Forever Blueshirts, Minnesota Sports Fan, LAFB Network, Bounding Into Sports, and Ringside Intel.

- `get_lifestyle_news_recommendations`: A tool that fetches Real-time updates, analysis, and personalized content from top sources like The Mix, Snipdaily, Nerdable, and Familyproof.

- `get_iheartdogs_recommendations`: A tool that fetches articles on health, behavior, lifestyle. grooming, ownership and more from iheartdogs.com

- `get_iheartcats_recommendations`: A tool that fetches articles on health, behavior, lifestyle. grooming, ownership and more from iheartcats.com

- `get_greenmonster_recommendations`: A tool that fetches guides to making conscious and compassionate choices that help people, animals, and the planet.

- `get_wishtv_recommendations`: A tool that fetches politics, breaking news, multicultural news, Hispanic language content, Entertainment, Health, Education and many more.

- `get_nine_and_ten_news_recommendations`: A tool that fetches up-to-date local news, weather forecasts, sports coverage, and community stories for Northern Michigan, including the Cadillac and Traverse City areas.

This loader is designed to be used as a way to load data as a Tool in an Agent.
