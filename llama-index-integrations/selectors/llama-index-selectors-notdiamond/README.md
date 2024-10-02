# LlamaIndex Selectors Integration: NotDiamond

[Not Diamond](https://notdiamond.ai) offers an AI-powered model router that automatically determines which LLM is best suited to respond to any query, improving LLM output quality by combining multiple LLMs into a **meta-model** that learns when to call each LLM.

Not Diamond supports cost and latency tradeoffs, customized router training, and real-time router personalization. Learn more via [the documentation](https://notdiamond.readme.io/).

## Installation

```shell
pip install llama-index-selectors-notdiamond
```

## Configuration

- API keys: You need API keys from [Not Diamond](https://app.notdiamond.ai/keys) and any LLM you want to use.

## Quick Start

```python
import os
from typing import List

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
    Settings,
)
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.selectors.notdiamond.base import NotDiamondSelector

from notdiamond import NotDiamond


# Set up your API keys
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
os.environ["NOTDIAMOND_API_KEY"] = "sk-..."


# Create indexes
documents = SimpleDirectoryReader("data/paul_graham").load_data()
nodes = Settings.node_parser.get_nodes_from_documents(documents)

vector_index = VectorStoreIndex.from_documents(documents)
summary_index = SummaryIndex.from_documents(documents)
query_text = "What was Paul Graham's role at Yahoo?"


# Set up Tools for the QueryEngine
list_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()

list_tool = QueryEngineTool.from_defaults(
    query_engine=list_query_engine,
    description=(
        "Useful for summarization questions related to Paul Graham eassy on"
        " What I Worked On."
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from Paul Graham essay on What"
        " I Worked On."
    ),
)


# Create a NotDiamondSelector and RouterQueryEngine
client = NotDiamond(
    api_key=os.environ["NOTDIAMOND_API_KEY"],
    llm_configs=["openai/gpt-4o", "anthropic/claude-3-5-sonnet-20240620"],
)
preference_id = client.create_preference_id()
client.preference_id = preference_id

nd_selector = NotDiamondSelector(client=client)

query_engine = RouterQueryEngine(
    selector=nd_selector,
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)


# Use Not Diamond to Query Indexes
response = query_engine.query(
    "Please summarize Paul Graham's working experience."
)
print(str(response))
```
