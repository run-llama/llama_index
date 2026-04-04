---
title: Hindsight
---

[Hindsight](https://github.com/vectorize-io/hindsight) is an open-source (MIT) long-term memory engine for AI agents. It automatically extracts facts from conversations, builds entity graphs, and retrieves relevant context using four parallel strategies (semantic, BM25, graph traversal, temporal).

The `hindsight-llamaindex` package provides two integration patterns:

- **Tools** (`HindsightToolSpec`): Agent-driven memory via `BaseToolSpec`. The agent decides when to retain, recall, or reflect.
- **Memory** (`HindsightMemory`): Automatic memory via `BaseMemory`. Messages are stored on every turn and recalled as context.

## Installation and setup

```bash
pip install hindsight-llamaindex
```

Hindsight runs locally via Docker, pip, or embedded in Python. See the [quick start guide](https://github.com/vectorize-io/hindsight#quick-start) for setup instructions.

## Agent tools

Give your agent retain/recall/reflect tools and let it decide when to use memory.

```python
import asyncio

from hindsight_client import Hindsight
from hindsight_llamaindex import HindsightToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI


async def main():
    client = Hindsight(base_url="http://localhost:8888")

    spec = HindsightToolSpec(
        client=client,
        bank_id="user-123",
        mission="Track user preferences",
    )
    tools = spec.to_tool_list()

    agent = ReActAgent(tools=tools, llm=OpenAI(model="gpt-4o"))

    response = await agent.run("Remember that I prefer dark mode")
    print(response)

    response = await agent.run("What do you know about my preferences?")
    print(response)


asyncio.run(main())
```

The agent gets three tools: `retain_memory` (store), `recall_memory` (search), and `reflect_on_memory` (synthesize). Use `create_hindsight_tools()` to select which tools to include.

## Automatic memory

For automatic memory without explicit tool calls, use `HindsightMemory`. On every turn, it retains messages to Hindsight and recalls relevant context before the LLM responds.

```python
import asyncio

from hindsight_client import Hindsight
from hindsight_llamaindex import HindsightMemory
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI


async def main():
    client = Hindsight(base_url="http://localhost:8888")

    memory = HindsightMemory.from_client(
        client=client,
        bank_id="user-123",
        mission="Track user preferences",
    )

    agent = ReActAgent(tools=[], llm=OpenAI(model="gpt-4o"))

    # Memory is passed per-run, not to the constructor
    response = await agent.run("I work at Acme Corp on the platform team", memory=memory)
    print(response)

    # On subsequent runs, memories from prior turns are recalled automatically
    response = await agent.run("What team do I work on?", memory=memory)
    print(response)


asyncio.run(main())
```

`HindsightMemory` maintains a local chat history buffer for the current session, while Hindsight provides cross-session long-term memory via semantic recall.

## Resources

- [Hindsight GitHub](https://github.com/vectorize-io/hindsight)
- [hindsight-llamaindex on PyPI](https://pypi.org/project/hindsight-llamaindex/)
- [Integration documentation](https://docs.hindsight.vectorize.io/sdks/integrations/llamaindex)
- [Configuration and advanced usage](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/llamaindex)
