# Seltz Web Knowledge Tool

[Seltz](https://www.seltz.ai/) provides fast, up-to-date web data with context-engineered web content and sources for real-time AI reasoning. Web content is processed and shaped to maximize usefulness for LLMs, AI agents, and RAG pipelines.

To begin, you need to obtain an API key from [Seltz](https://www.seltz.ai/).

## Installation

```bash
pip install llama-index-tools-seltz
```

## Usage

```python
from llama_index.tools.seltz import SeltzToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

seltz_tool = SeltzToolSpec(api_key="your-seltz-api-key")

agent = FunctionAgent(
    tools=seltz_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
)

await agent.run("What are the latest developments in AI reasoning?")
```

## Available Functions

`search`: Search the web using Seltz and return relevant documents with sources. Returns a list of Document objects containing web content and source URLs.

### Parameters

- `query` (str): The search query text.
- `max_documents` (int, optional): Maximum number of documents to return (default: 10).

### Example

```python
from llama_index.tools.seltz import SeltzToolSpec

seltz_tool = SeltzToolSpec(api_key="your-seltz-api-key")

documents = seltz_tool.search("web knowledge for AI agents", max_documents=5)

for doc in documents:
    print(f"URL: {doc.metadata['url']}")
    print(f"Content: {doc.text[:200]}...")
```

This tool is designed to be used as a way to load data as a Tool in an Agent.
