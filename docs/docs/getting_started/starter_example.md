# Starter Tutorial (Using OpenAI)

This tutorial will show you how to get started building agents with LlamaIndex. We'll start with a basic example and then show how to add RAG (Retrieval-Augmented Generation) capabilities.

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

!!! tip
    If you are using an OpenAI-Compatible API, you can use the `OpenAILike` LLM class. You can find more information in the [OpenAILike LLM](https://docs.llamaindex.ai/en/stable/api_reference/llms/openai_like/) integration and [OpenAILike Embeddings](https://docs.llamaindex.ai/en/stable/api_reference/embeddings/openai_like/) integration.

## Basic Agent Example

Let's start with a simple example using an agent that can perform basic multiplication by calling a tool. Create a file called `starter.py`:

```python
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI


# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    tools=[multiply],
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

This will output something like: `The result of \( 1234 \times 4567 \) is \( 5,678,678 \).`

What happened is:

- The agent was given a question: `What is 1234 * 4567?`
- Under the hood, this question, plus the schema of the tools (name, docstring, and arguments) were passed to the LLM
- The agent selected the `multiply` tool and wrote the arguments to the tool
- The agent received the result from the tool and interpolated it into the final response

!!! tip
    As you can see, we are using `async` python functions. Many LLMs and models support async calls, and using async code is recommended to improve performance of your application. To learn more about async code and python, we recommend this [short section on async + python](./async_python.md).

## Adding Chat History

The `AgentWorkflow` is also able to remember previous messages. This is contained inside the `Context` of the `AgentWorkflow`.

If the `Context` is passed in, the agent will use it to continue the conversation.

```python
from llama_index.core.workflow import Context

# create context
ctx = Context(agent)

# run agent with context
response = await agent.run("My name is Logan", ctx=ctx)
response = await agent.run("What is my name?", ctx=ctx)
```

## Adding RAG Capabilities

Now let's enhance our agent by adding the ability to search through documents. First, let's get some example data using our terminal:

```bash
mkdir data
wget https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt -O data/paul_graham_essay.txt
```

Your directory structure should look like this now:

<pre>
├── starter.py
└── data
    └── paul_graham_essay.txt
</pre>

Now we can create a tool for searching through documents using LlamaIndex. By default, our `VectorStoreIndex` will use a `text-embedding-ada-002` embeddings from OpenAI to embed and retrieve the text.

Our modified `starter.py` should look like this:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
import asyncio
import os

# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about an personal essay written by Paul Graham."""
    response = await query_engine.aquery(query)
    return str(response)


# Create an enhanced workflow with both tools
agent = FunctionAgent(
    tools=[multiply, search_documents],
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
    If you used a [vector store integration](../module_guides/storing/vector_stores.md) besides the default, chances are you can just reload from the vector store:

    ```python
    index = VectorStoreIndex.from_vector_store(vector_store)
    ```

## What's Next?

This is just the beginning of what you can do with LlamaIndex agents! You can:

- Add more tools to your agent
- Use different LLMs
- Customize the agent's behavior using system prompts
- Add streaming capabilities
- Implement human-in-the-loop workflows
- Use multiple agents to collaborate on tasks

Some helpful next links:

- See more advanced agent examples in our [Agent documentation](../understanding/agent/index.md)
- Learn more about [high-level concepts](./concepts.md)
- Explore how to [customize things](./customization.md)
- Check out the [component guides](../module_guides/index.md)
