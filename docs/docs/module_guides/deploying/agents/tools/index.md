# Tools

## Concept

Having proper tool abstractions is at the core of building [data agents](../index.md). Defining a set of Tools is similar to defining any API interface, with the exception that these Tools are meant for agent rather than human use. We allow users to define both a **Tool** as well as a **ToolSpec** containing a series of functions under the hood.

A Tool implements a very generic interface - simply define `__call__` and also return some basic metadata (name, description, function schema).

A Tool Spec defines a full API specification of any service that can be converted into a list of Tools.

We offer a few different types of Tools:

- `FunctionTool`: A function tool allows users to easily convert any user-defined function into a Tool. It can also auto-infer the function schema.
- `QueryEngineTool`: A tool that wraps an existing [query engine](../../query_engine/index.md). Note: since our agent abstractions inherit from `BaseQueryEngine`, these tools can also wrap other agents.

We offer a rich set of Tools and Tool Specs through [LlamaHub](https://llamahub.ai/) 🦙.

### Blog Post

For full details, please check out our detailed [blog post](https://blog.llamaindex.ai/building-better-tools-for-llm-agents-f8c5a6714f11).

## Usage Pattern

Our Tool Specs and Tools can be imported from the `llama-hub` package.

To use with our agent,

```python
from llama_index.agent.openai import OpenAIAgent
from llama_index.tools.google import GmailToolSpec

tool_spec = GmailToolSpec()
agent = OpenAIAgent.from_tools(tool_spec.to_tool_list(), verbose=True)
```

See our [Usage Pattern Guide](./usage_pattern.md) for more details.

## LlamaHub Tools Guide 🛠️

Check out [our guide](./llamahub_tools_guide.md) for a full overview of the Tools/Tool Specs in LlamaHub!
