# Gestell SDK Pack

A LlamaIndex Pack for integrating Gestell SDK’s search and prompt endpoints as tool calls.

## Installation

Download this pack using the LlamaIndex CLI:

```bash
llamaindex-cli download-llamapack GestellSDKPack --download-dir ./gestell_pack
```

### Alternatively, download via python

```python
from llama_index.core.llama_pack import download_llama_pack

# Download the pack to ./gestell_pack

PackClass = download_llama_pack("GestellSDKPack", "./gestell_pack")
```

## Quickstart

Make sure you have your environment properly set:

1. `GESTELL_API_KEY`: Your Gestell SDK API key
2. `GESTELL_COLLECTION_ID`: Default collection id for queries, this can be overridden dynamically in tool calls.

### Prompting

```python
from llama_index.packs.gestell_sdk import GestellSDKPack

pack = GestellSDKPack(
    api_key="your_api_key",  # Optional if GESTELL_API_KEY is set
    collection_id="your_collection_id",  # Optional if GESTELL_COLLECTION_ID is set
)

response = pack.run("Summarize the documents")
print(response)
```

### Asynchronous Prompting

```python
# ... imports


async def main():
    async for chunk in pack.aprompt("Summarize the documents"):
        print(chunk, end="", flush=True)


asyncio.run(main())
```

### Search Queries

```python
# ... imports


async def search_example():
    results = await pack.asearch("What are the key findings from the 10-Q?")
    for i, hit in enumerate(results, start=1):
        print(f"[{i}] Citation: {hit['citation']}")
        print(f"Preview: {hit['content'][:200]}...\n")


asyncio.run(search_example())
```

## Usage in Agents

Extract `FunctionTool` definitions for LlamaIndex agents:

```python
tools = pack.get_tools()

# - 'gestell_search': returns JSON search results
# - 'gestell_prompt': streams text responses directly
```
