# LlamaIndex Tools Integration: SERPEX

This tool allows you to use SERPEX API to search the web and get real-time results from multiple search engines within your LlamaIndex application.

## Installation

```bash
pip install llama-index-tools-serpex
```

## Usage

```python
from llama_index.tools.serpex import SerpexToolSpec
from llama_index.agent.openai import OpenAIAgent

# Initialize the tool
serpex_tool = SerpexToolSpec(api_key="your_serpex_api_key")

# Create agent with the tool
agent = OpenAIAgent.from_tools(serpex_tool.to_tool_list(), verbose=True)

# Use the agent
response = agent.chat("What are the latest AI developments?")
print(response)
```

### Advanced Usage

```python
# Use specific search engine
serpex_tool = SerpexToolSpec(
    api_key="your_api_key",
    engine="google",  # or 'bing', 'duckduckgo', 'brave', etc.
)

# Search with time filter
results = serpex_tool.search(
    "recent AI news",
    num_results=10,
    time_range="day",  # 'day', 'week', 'month', 'year'
)

# Use different engines for different queries
results = serpex_tool.search(
    "privacy tools", engine="duckduckgo", num_results=5
)
```

## API Key

Get your API key from [SERPEX Dashboard](https://serpex.dev/dashboard).

Set as environment variable:

```bash
export SERPEX_API_KEY=your_api_key
```

## Features

- **Multiple Search Engines**: Auto-routing, Google, Bing, DuckDuckGo, Brave, Yahoo, Yandex
- **Real-time Results**: Get up-to-date search results via API
- **Time Filtering**: Filter by day, week, month, or year
- **Fast & Reliable**: 99.9% uptime SLA with global proxy network
- **Structured Data**: Clean JSON responses optimized for AI applications
- **Cost Effective**: Only 1 credit per request, failed requests free

## Search Engines

- `auto` - Automatically routes to the best available engine (default)
- `google` - Google Search
- `bing` - Microsoft Bing
- `duckduckgo` - Privacy-focused DuckDuckGo
- `brave` - Brave Search
- `yahoo` - Yahoo Search
- `yandex` - Yandex Search

## Links

- [SERPEX Website](https://serpex.dev)
- [SERPEX Documentation](https://serpex.dev/docs)
- [SERPEX Dashboard](https://serpex.dev/dashboard)
- [LlamaIndex](https://llamaindex.ai)
