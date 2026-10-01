# LlamaIndex Tools Integration: SERPEX

This tool lets your LlamaIndex agents search the web with Serpex.

Serpex is a web search API and extract API for AI agents. Search returns ranked web results, optionally with page content as markdown; Extract turns known URLs into clean markdown. Serpex runs its own search engine.

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
serpex_tool = SerpexToolSpec(api_key="your_api_key")

# Search with time filter
results = serpex_tool.search(
    "recent AI news",
    num_results=10,
    time_range="day",  # 'day', 'week', 'month', 'year'
)
```

## API Key

Get your API key from [SERPEX Dashboard](https://serpex.dev/dashboard).

Set as environment variable:

```bash
export SERPEX_API_KEY=your_api_key
```

## Features

- **Ranked Web Results**: Title, URL and snippet for each result, returned as LlamaIndex `Document`s
- **Time Filtering**: Filter by day, week, month, or year
- **Structured Data**: Clean JSON responses for AI applications

## The `engine` parameter (deprecated)

`engine` is deprecated and ignored. Results are routed automatically. It is still
accepted so existing code keeps working.

## Links

- [SERPEX Website](https://serpex.dev)
- [SERPEX Documentation](https://serpex.dev/docs)
- [SERPEX Dashboard](https://serpex.dev/dashboard)
- [LlamaIndex](https://llamaindex.ai)
