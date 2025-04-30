# Chain-of-Abstraction Agent Pack

`pip install llama-index-packs-agents-coa`

The chain-of-abstraction (CoA) LlamaPack implements a generalized version of the strategy described in the [origin CoA paper](https://arxiv.org/abs/2401.17464).

By prompting the LLM to write function calls in a chain-of-thought format, we can execute both simple and complex combinations of function calls needed to execute a task.

The LLM is prompted to write a response containing function calls, for example, a CoA plan might look like:

```
After buying the apples, Sally has [FUNC add(3, 2) = y1] apples.
Then, the wizard casts a spell to multiply the number of apples by 3,
resulting in [FUNC multiply(y1, 3) = y2] apples.
```

From there, the function calls can be parsed into a dependency graph, and executed.

Then, the values in the CoA are replaced with their actual results.

As an extension to the original paper, we also run the LLM a final time, to rewrite the response in a more readable and user-friendly way.

**NOTE:** In the original paper, the authors fine-tuned an LLM specifically for this, and also for specific functions and datasets. As such, only capabale LLMs (OpenAI, Anthropic, etc.) will be (hopefully) reliable for this without finetuning.

## Code Usage

`pip install llama-index-packs-agents-coa`

First, setup some tools (could be function tools, query engines, etc.)

```python
from llama_index.core.tools import QueryEngineTool, FunctionTool


def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


query_engine = index.as_query_engine(...)

function_tool = FunctionTool.from_defaults(fn=add)
query_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine, name="...", description="..."
)
```

Next, create the pack with the tools, and run it!

```python
from llama_index.packs.agents_coa import CoAAgentPack
from llama_index.llms.openai import OpenAI

pack = CoAAgentPack(
    tools=[function_tool, query_tool], llm=OpenAI(model="gpt-4")
)

print(pack.run("What is 1245 + 4321?"))
```
