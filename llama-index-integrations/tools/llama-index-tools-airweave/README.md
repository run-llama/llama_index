# LlamaIndex Tools Integration: Airweave

This tool connects your LlamaIndex agent to [Airweave](https://airweave.ai/), an open-source platform that makes any app searchable by syncing data from various sources with minimal configuration.

## Installation

```bash
pip install llama-index-tools-airweave llama-index-llms-openai
```

## Prerequisites

1. An Airweave account and API key
2. At least one collection set up with synced data

Get started at [Airweave](https://airweave.ai/)

## Usage

### Basic Usage

```python
import os
import asyncio
from llama_index.tools.airweave import AirweaveToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

# Initialize the Airweave tool
airweave_tool = AirweaveToolSpec(
    api_key=os.environ["AIRWEAVE_API_KEY"],
)

# Create an agent with the Airweave tools
agent = FunctionAgent(
    tools=airweave_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="""You are a helpful assistant that can search through
    Airweave collections to answer questions about your organization's data.""",
)


# Use the agent to search your data
async def main():
    response = await agent.run(
        "Search the finance-data collection for Q4 revenue reports"
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
```

## Available Tools

### `search_collection`

Simple search in a collection with default settings (most common use case).

**Parameters:**

- `collection_id` (str): The readable ID of the collection
- `query` (str): Your search query
- `limit` (int, optional): Max results to return (default: 10)
- `offset` (int, optional): Pagination offset (default: 0)

### `advanced_search_collection`

Advanced search with full control over retrieval parameters.

**Parameters:**

- `collection_id` (str): The readable ID of the collection
- `query` (str): Your search query
- `limit` (int, optional): Max results to return (default: 10)
- `offset` (int, optional): Pagination offset (default: 0)
- `retrieval_strategy` (str, optional): "hybrid", "neural", or "keyword"
- `temporal_relevance` (float, optional): Weight recent content (0.0-1.0)
- `expand_query` (bool, optional): Generate query variations
- `interpret_filters` (bool, optional): Extract filters from natural language
- `rerank` (bool, optional): Use LLM-based reranking
- `generate_answer` (bool, optional): Generate natural language answer

**Returns:**
Dictionary with `documents` list and optional `answer` field.

### `search_and_generate_answer`

Convenience method that searches and returns a direct natural language answer (RAG-style).

**Parameters:**

- `collection_id` (str): The readable ID of the collection
- `query` (str): Your question in natural language
- `limit` (int, optional): Max results to consider (default: 10)
- `use_reranking` (bool, optional): Use reranking (default: True)

**Returns:**
Natural language answer string.

### `list_collections`

List all collections in your organization.

**Parameters:**

- `skip` (int, optional): Pagination skip (default: 0)
- `limit` (int, optional): Max collections to return (default: 100)

### `get_collection_info`

Get detailed information about a specific collection.

**Parameters:**

- `collection_id` (str): The readable ID of the collection

## Advanced Examples

### Direct Tool Usage

You can use the tools directly without an agent:

```python
from llama_index.tools.airweave import AirweaveToolSpec

airweave_tool = AirweaveToolSpec(api_key="your-key")

# List collections
collections = airweave_tool.list_collections()
print(f"Found {len(collections)} collections")

# Simple search
results = airweave_tool.search_collection(
    collection_id="finance-data", query="Q4 revenue reports", limit=5
)

for doc in results:
    print(f"Score: {doc.metadata.get('score', 'N/A')}")
    print(f"Text: {doc.text[:200]}...")
```

### Advanced Search Options

```python
# Advanced search with all options
result = airweave_tool.advanced_search_collection(
    collection_id="finance-data",
    query="Q4 revenue reports",
    limit=20,
    retrieval_strategy="hybrid",  # hybrid, neural, or keyword
    temporal_relevance=0.3,  # Weight recent content (0.0-1.0)
    expand_query=True,  # Query expansion for better recall
    interpret_filters=True,  # Extract filters from natural language
    rerank=True,  # LLM reranking for better relevance
    generate_answer=True,  # Generate natural language answer
)

# Access results
documents = result["documents"]
if "answer" in result:
    print(f"Generated Answer: {result['answer']}")
```

### RAG-Style Direct Answers

```python
# Get a direct answer instead of raw documents
answer = airweave_tool.search_and_generate_answer(
    collection_id="finance-data",
    query="What was our Q4 revenue growth?",
    limit=10,
    use_reranking=True,
)
print(answer)  # "Q4 revenue grew by 23% to $45M compared to Q3..."
```

### Using Different Retrieval Strategies

```python
# Keyword search for exact term matching
results = airweave_tool.advanced_search_collection(
    collection_id="legal-docs",
    query="GDPR compliance",
    retrieval_strategy="keyword",  # Use BM25 keyword search
)

# Neural search for semantic understanding
results = airweave_tool.advanced_search_collection(
    collection_id="research-papers",
    query="papers about transformer architectures",
    retrieval_strategy="neural",  # Pure semantic search
)

# Hybrid search (default) - best of both worlds
results = airweave_tool.advanced_search_collection(
    collection_id="all-docs",
    query="machine learning best practices",
    retrieval_strategy="hybrid",  # Combines semantic + keyword
)
```

### Temporal Relevance

Weight recent documents higher in results:

```python
# Strongly prefer recent content
results = airweave_tool.advanced_search_collection(
    collection_id="news-articles",
    query="AI breakthroughs",
    temporal_relevance=0.8,  # 0.0 = no recency bias, 1.0 = only recent matters
)
```

### Agent with Advanced Search

Agents can automatically leverage these features:

```python
agent = FunctionAgent(
    tools=airweave_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="""You have access to advanced Airweave search capabilities:
    - Use search_collection for simple queries
    - Use advanced_search_collection when you need temporal filtering, reranking, etc.
    - Use search_and_generate_answer to get direct answers from documents

    When searching recent information, use temporal_relevance.
    When you need precise answers, use search_and_generate_answer.
    """,
)


async def main():
    response = await agent.run(
        "Search for recent updates in the engineering-docs collection and summarize them"
    )
    print(response)


asyncio.run(main())
```

## Custom Base URL

If you're self-hosting Airweave:

```python
airweave_tool = AirweaveToolSpec(
    api_key="your-api-key",
    base_url="https://your-airweave-instance.com",
)
```

## Using with Local Models

If you want to use local models instead of OpenAI:

```python
from llama_index.llms.ollama import Ollama

agent = FunctionAgent(
    tools=airweave_tool.to_tool_list(),
    llm=Ollama(model="llama3.1", request_timeout=360.0),
)
```

## Learn More

- [Airweave Documentation](https://docs.airweave.ai/)
- [Airweave GitHub](https://github.com/airweave-ai/airweave)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This integration is released under the MIT License.
