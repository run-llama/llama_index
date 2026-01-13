# Parallel AI Tool

This tool provides integration between LlamaIndex and [Parallel AI](https://parallel.ai/)'s Search and Extract APIs, enabling LLM agents to perform web research and content extraction.

- **Search API**: Returns structured, compressed excerpts from web search results optimized for LLM consumption
- **Extract API**: Converts public URLs into clean, LLM-optimized markdown including JavaScript-heavy pages and PDFs

## Installation

```bash
pip install llama-index-tools-parallel-web-systems
```

## Setup

1. Get your API key from [Parallel AI Platform](https://platform.parallel.ai/)
2. Set your API key as an environment variable or pass it directly

## Usage

```python
from llama_index.tools.parallel_web_systems import ParallelWebSystemsToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

# Initialize the tool with your API key
parallel_tool = ParallelWebSystemsToolSpec(
    api_key="your-api-key-here",
)

# Create an agent with the tool
agent = FunctionAgent(
    tools=parallel_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
)

# Use the agent to perform web research
response = await agent.run("What was the GDP of France in 2023?")
print(response)
```

## Available Functions

### `search`

Search the web using Parallel AI's Search API. Returns structured excerpts optimized for LLM consumption.

**Parameters:**

- `objective` (str, optional): Natural-language description of what to search for
- `search_queries` (list[str], optional): Traditional keyword search queries (max 5)
- `max_results` (int): Maximum results to return, 1-40 (default: 10)
- `mode` (str, optional): `'one-shot'` for comprehensive results, `'agentic'` for token-efficient results
- `excerpts` (dict, optional): Excerpt settings, e.g., `{'max_chars_per_result': 1500}`
- `source_policy` (dict, optional): Domain and date preferences
- `fetch_policy` (dict, optional): Cache vs live content policy

At least one of `objective` or `search_queries` must be provided.

**Example:**

```python
from llama_index.tools.parallel_web_systems import ParallelWebSystemsToolSpec

parallel_tool = ParallelWebSystemsToolSpec(api_key="your-api-key")

# Search with an objective
results = parallel_tool.search(
    objective="What are the latest developments in renewable energy?",
    max_results=5,
    mode="one-shot",
)

for doc in results:
    print(f"Title: {doc.metadata.get('title')}")
    print(f"URL: {doc.metadata.get('url')}")
    print(f"Excerpts: {doc.text[:300]}...")
    print("---")

# Search with specific queries
results = parallel_tool.search(
    search_queries=["solar power 2024", "wind energy statistics"],
    max_results=8,
    mode="agentic",
)
```

### `extract`

Extract clean, structured content from web pages using Parallel AI's Extract API.

**Parameters:**

- `urls` (list[str]): List of URLs to extract content from
- `objective` (str, optional): Natural language objective to focus extraction
- `search_queries` (list[str], optional): Specific keyword queries to focus extraction
- `excerpts` (bool | dict): Include excerpts (default: True). Can be dict like `{'max_chars_per_result': 2000}`
- `full_content` (bool | dict): Include full page content (default: False)
- `fetch_policy` (dict, optional): Cache vs live content policy

**Example:**

```python
from llama_index.tools.parallel_web_systems import ParallelWebSystemsToolSpec

parallel_tool = ParallelWebSystemsToolSpec(api_key="your-api-key")

# Extract content focused on a specific objective
results = parallel_tool.extract(
    urls=["https://en.wikipedia.org/wiki/Artificial_intelligence"],
    objective="What are the main applications and ethical concerns of AI?",
    excerpts={"max_chars_per_result": 2000},
)

for doc in results:
    print(f"Title: {doc.metadata.get('title')}")
    print(f"Content: {doc.text[:500]}...")

# Extract full content from multiple URLs
results = parallel_tool.extract(
    urls=[
        "https://example.com/article1",
        "https://example.com/article2",
    ],
    full_content=True,
    excerpts=False,
)
```

## Error Handling

The tool includes built-in error handling. If an API call fails, it returns an empty list, allowing your agent to continue:

```python
results = parallel_tool.search(objective="test query")
if not results:
    print("No results found or API error occurred")
```

For extract operations, failed URLs are included in results with error information:

```python
results = parallel_tool.extract(urls=["https://invalid-url.com/"])
for doc in results:
    if doc.metadata.get("error_type"):
        print(f"Failed: {doc.metadata['url']} - {doc.text}")
```

## License

MIT
