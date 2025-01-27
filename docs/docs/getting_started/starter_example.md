# Starter Tutorial

This tutorial will show you how to get started with LlamaIndex using our Agent capabilities. We'll start with a basic example and then show how to add RAG (Retrieval-Augmented Generation) capabilities.

!!! tip
    Make sure you've followed the [installation](installation.md) steps first.

!!! tip
    Want to use local models?
    If you want to do our starter tutorial using only local models, [check out this tutorial instead](starter_example_local.md).

## Set your OpenAI API key

LlamaIndex uses OpenAI's `gpt-3.5-turbo` by default. Make sure your API key is available to your code by setting it as an environment variable:

```bash
# MacOS/Linux
export OPENAI_API_KEY=XXXXX

# Windows
set OPENAI_API_KEY=XXXXX
```

## Basic Agent Example

Let's start with a simple example using an agent that can perform basic multiplication. Create a file called `starter.py`:

```python
import asyncio
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.openai import OpenAI


# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


# Create an agent workflow with our calculator tool
agent = AgentWorkflow.from_tools_or_functions(
    [multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)


async def main():
    # Run the agent
    response = await agent.run("What is 1234 * 4567?")
    print(str(response))


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
```

This will output something like: `The result of 123 * 456 is 56,088.`

## Adding Chat History

The `AgentWorkflow` is also able to remember previous messages. This is contained inside the `Context` of the `AgentWorkflow`.

If the `Context` is passed in, the agent will use it to continue the conversation.

```python
from llama_index.core.context import Context

# create context
ctx = Context(agent)

# run agent with context
response = await agent.run("My name is Logan", ctx=ctx)
response = await agent.run("What is my name?", ctx=ctx)
```

## Adding RAG Capabilities

Now let's enhance our agent by adding the ability to search through documents. First, let's get some example data:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os

# Download example data if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")
    # Download Paul Graham essay as an example
    import requests

    url = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
    response = requests.get(url)
    with open("data/paul_graham_essay.txt", "w") as f:
        f.write(response.text)

# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()


async def search_documents(query: str) -> str:
    """Search through documents to find relevant information."""
    return str(query_engine.query(query))


# Create an enhanced workflow with both tools
agent = AgentWorkflow.from_tools_or_functions(
    [calculate, search_documents],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="""You are a helpful assistant that can perform calculations
    and search through documents to answer questions.""",
)


# Now we can ask questions about the documents or do calculations
async def main():
    response = await agent.run(
        "What did the author do in college? Also, what's 7 * 8?"
    )
    print(response)


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
```

The agent can now seamlessly switch between using the calculator and searching through documents to answer questions.

## Storing the RAG Index

To avoid reprocessing documents every time, you can persist the index to disk:

```python
# Save the index
index.storage_context.persist("storage")

# Later, load the index
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()
```

!!! tip
    If you used a vector store integration besides the default, chances are you can just reload from the vector store:

    ```python
    index = VectorStoreIndex.from_vector_store(vector_store)
    ```

## What's Next?

This is just the beginning of what you can do with LlamaIndex agents! You can:

- Add more tools to your agent
- Use different LLMs
- Customize the agent's behavior
- Add streaming capabilities
- Implement human-in-the-loop workflows
- Use multiple agents to collaborate on tasks

!!! tip
    - See more advanced agent examples in our [Agent documentation](../understanding/agent/multi_agents.md)
    - Learn more about [high-level concepts](./concepts.md)
    - Explore how to [customize things](./customization.md)
    - Check out the [component guides](../module_guides/index.md)
