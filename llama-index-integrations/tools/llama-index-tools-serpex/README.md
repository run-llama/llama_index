# LlamaIndex Tools Integration: Serpex

Search the web from a LlamaIndex agent with [Serpex](https://serpex.dev), a real-time web search API for AI agents and data enrichment. Each search returns ranked web results (title, URL and snippet) as LlamaIndex `Document` objects.

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

### Search directly

```python
# Search with a time filter
results = serpex_tool.search(
    "recent AI news",
    num_results=10,
    time_range="day",  # 'day', 'week', 'month', 'year'
)

for doc in results:
    print(doc.metadata["title"], doc.metadata["url"])
```

The `engine` argument is deprecated. It is still accepted so existing code keeps working, but the Serpex API ignores it and routes every search itself.

## API Key

Get your API key from the [Serpex dashboard](https://app.serpex.dev/dashboard).

Set it as an environment variable:

```bash
export SERPEX_API_KEY=your_api_key
```

## Features

- **Real-time results**: live web results for every query
- **Time filtering**: day, week, month or year
- **Structured data**: clean JSON, returned as LlamaIndex `Document` objects
- **Pay per use**: 1 credit per search; failed requests are not charged

## Links

- [Serpex website](https://serpex.dev)
- [Serpex documentation](https://serpex.dev/docs)
- [LlamaIndex](https://llamaindex.ai)
