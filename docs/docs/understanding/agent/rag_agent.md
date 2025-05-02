# Adding RAG to an agent

To demonstrate using RAG engines as a tool in an agent, we're going to create a very simple RAG query engine. Our source data is going to be the [Wikipedia page about the 2023 Canadian federal budget](https://en.wikipedia.org/wiki/2023_Canadian_federal_budget) that we've [printed as a PDF](https://www.dropbox.com/scl/fi/rop435rax7mn91p3r8zj3/2023_canadian_budget.pdf?rlkey=z8j6sab5p6i54qa9tr39a43l7&dl=0).

## Bring in new dependencies

To read the PDF and index it, we'll need a few new dependencies. They were installed along with the rest of LlamaIndex, so we just need to import them:

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
```

## Add LLM to settings

We were previously passing the LLM directly, but now we need to use it in multiple places, so we'll add it to the global settings.

```python
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
```

Place this line near the top of the file; you can delete the other `llm` assignment.

## Load and index documents

We'll now do 3 things in quick succession: we'll load the PDF from a folder called "data", index and embed it using the `VectorStoreIndex`, and then create a query engine from that index:

```python
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
```

We can run a quick smoke-test to make sure the engine is working:

```python
response = query_engine.query(
    "What was the total amount of the 2023 Canadian federal budget?"
)
print(response)
```

The response is fast:

```
The total amount of the 2023 Canadian federal budget was $496.9 billion.
```

## Add a query engine tool

This requires one more import:

```python
from llama_index.core.tools import QueryEngineTool
```

Now we turn our query engine into a tool by supplying the appropriate metadata (for the python functions, this was being automatically extracted so we didn't need to add it):

```python
budget_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="canadian_budget_2023",
    description="A RAG engine with some basic facts about the 2023 Canadian federal budget.",
)
```

We modify our agent by adding this engine to our array of tools (we also remove the `llm` parameter, since it's now provided by settings):

```python
agent = ReActAgent.from_tools(
    [multiply_tool, add_tool, budget_tool], verbose=True
)
```

## Ask a question using multiple tools

This is kind of a silly question, we'll ask something more useful later:

```python
response = agent.chat(
    "What is the total amount of the 2023 Canadian federal budget multiplied by 3? Go step by step, using a tool to do any math."
)

print(response)
```

We get a perfect answer:

```
Thought: The current language of the user is English. I need to use the tools to help me answer the question.
Action: canadian_budget_2023
Action Input: {'input': 'total'}
Observation: $496.9 billion
Thought: I need to multiply the total amount of the 2023 Canadian federal budget by 3.
Action: multiply
Action Input: {'a': 496.9, 'b': 3}
Observation: 1490.6999999999998
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: The total amount of the 2023 Canadian federal budget multiplied by 3 is $1,490.70 billion.
The total amount of the 2023 Canadian federal budget multiplied by 3 is $1,490.70 billion.
```

As usual, you can check the [repo](https://github.com/run-llama/python-agents-tutorial/blob/main/3_rag_agent.py) to see this code all together.

Excellent! Your agent can now use any arbitrarily advanced query engine to help answer questions. You can also add as many different RAG engines as you need to consult different data sources. Next, we'll look at how we can answer more advanced questions [using LlamaParse](./llamaparse.md).
