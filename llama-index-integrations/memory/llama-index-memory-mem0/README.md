# LlamaIndex Memory Integration: Mem0

## Installation

To install the required package, run:

```bash
%pip install llama-index-memory-mem0
```

## Setup with Mem0 Platform

1. Set your Mem0 Platform API key as an environment variable. You can replace `<your-mem0-api-key>` with your actual API key:

> Note: You can obtain your Mem0 Platform API key from the [Mem0 Platform](https://app.mem0.ai/login).

```python
os.environ["MEM0_API_KEY"] = "<your-mem0-api-key>"
```

2. Import the necessary modules and create a Mem0Memory instance:

```python
from llama_index.memory.mem0 import Mem0Memory

context = {"user_id": "user_1"}
memory = Mem0Memory.from_client(
    context=context,
    api_key="<your-mem0-api-key>",
    search_msg_limit=4,  # optional, default is 5
)
```

Mem0 Context is used to identify the user, agent or the conversation in the Mem0. It is required to be passed in the at least one of the fields in the `Mem0Memory` constructor. It can be any of the following:

```python
context = {
    "user_id": "user_1",
    "agent_id": "agent_1",
    "run_id": "run_1",
}
```

`search_msg_limit` is optional, default is 5. It is the number of messages from the chat history to be used for memory retrieval from Mem0. More number of messages will result in more context being used for retrieval but will also increase the retrieval time and might result in some unwanted results.

## Setup with Mem0 OSS

1. Set your Mem0 OSS by providing configuration details:

> Note: To know more about Mem0 OSS, read [Mem0 OSS Quickstart](https://docs.mem0.ai/open-source/quickstart).

```python
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test_9",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 1536,  # Change this according to your local model's dimensions
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o",
            "temperature": 0.2,
            "max_tokens": 1500,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"},
    },
    "version": "v1.1",
}
```

2. Create a Mem0Memory instance:

```python
memory = Mem0Memory.from_config(
    context=context,
    config=config,
    search_msg_limit=4,  # optional, default is 5
)
```

## Basic Usage

Currently, Mem0 Memory is supported in the `SimpleChatEngine`, `FunctionCallingAgent` and `ReActAgent`.

Intilaize the LLM

```python
import os
from llama_index.llms.openai import OpenAI

os.environ["OPENAI_API_KEY"] = "<your-openai-api-key>"
llm = OpenAI(model="gpt-4o")
```

### SimpleChatEngine

```python
from llama_index.core import SimpleChatEngine

agent = SimpleChatEngine.from_defaults(
    llm=llm, memory=memory  # set you memory here
)

# Start the chat
response = agent.chat("Hi, My name is Mayank")
print(response)
```

Initialize the tools

```python
from llama_index.core.tools import FunctionTool


def call_fn(name: str):
    """Call the provided name.
    Args:
        name: str (Name of the person)
    """
    print(f"Calling... {name}")


def email_fn(name: str):
    """Email the provided name.
    Args:
        name: str (Name of the person)
    """
    print(f"Emailing... {name}")


call_tool = FunctionTool.from_defaults(fn=call_fn)
email_tool = FunctionTool.from_defaults(fn=email_fn)
```

### FunctionCallingAgent

```python
from llama_index.core.agent import FunctionCallingAgent

agent = FunctionCallingAgent.from_tools(
    [call_tool, email_tool],
    llm=llm,
    memory=memory,
    verbose=True,
)

# Start the chat
response = agent.chat("Hi, My name is Mayank")
print(response)
```

### ReActAgent

```python
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools(
    [call_tool, email_tool],
    llm=llm,
    memory=memory,
    verbose=True,
)

# Start the chat
response = agent.chat("Hi, My name is Mayank")
print(response)
```

> Note: For more examples refer to: [Notebooks](https://github.com/run-llama/llama_index/tree/main/docs/docs/examples/memory)

## References

- [Mem0 Platform](https://app.mem0.ai/login)
- [Mem0 OSS](https://docs.mem0.ai/open-source/quickstart)
- [Mem0 Github](https://github.com/mem0ai/mem0)
