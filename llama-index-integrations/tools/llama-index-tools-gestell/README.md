# Gestell Tool Spec

A LlamaIndex Tool Spec for integrating Gestell SDK’s search and prompt endpoints as tool calls.

## Quickstart

```python
import asyncio
from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.tools.gestell import GestellToolSpec


async def main():
    # 1. Load your .env, make sure OPENAI_API_KEY, GESTELL_API_KEY and GESTELL_COLLECTION_ID are available
    load_dotenv()

    # 2. Instantiate the Gestell tool spec
    tool_spec = GestellToolSpec()

    # 3. Generate the list of tools from that instance (can optionally add other tools for the agent to use)
    tool_list = tool_spec.to_tool_list()

    # 4. Create the OpenAI agent with these tools
    agent = OpenAIAgent.from_tools(tool_list, verbose=True)

    # 5. Run your queries
    response = await agent.achat(
        "Give me a concise summary of the documents in this collection."
    )
    print(response)


asyncio.run(main())
```

### Environment Configuration

Make sure you have your environment properly set:

1. `GESTELL_API_KEY`: Your Gestell SDK API key.
2. `GESTELL_COLLECTION_ID`: Default collection id for queries, this can be overridden dynamically in tool calls.

### Usage

1. Review [example usage](./examples/agent.py) with the ReAct agent for how it works with llama index agents.
2. Review [prompt usage](./examples//prompt.py) and [search usage](./examples/search.py) to see how the tools work directly
3. You can access the Gestell SDK client directly via `gestell.client`, [learn about usage here](https://gestell.ai/docs/reference).
