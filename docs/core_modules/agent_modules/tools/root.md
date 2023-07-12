# Tools

## Concept

Having proper tool abstractions is at the core of building data agents. Defining a set of Tools is similar to defining any API interface, with the exception that these Tools are meant for agent rather than human use. We allow users to define both a **Tool** as well as a **ToolSpec** containing a series of functions under the hood. 

A Tool implements a very generic interface - simply define `__call__` and also return some basic metadata (name, description, function schema).

A Tool Spec defines a full API specification of any service that can be converted into a list of Tools.

We offer a few different types of Tools:
- `FunctionTool`: A function tool allows users to easily convert any user-defined function into a Tool. It can also auto-infer the function schema.
- `QueryEngineTool`: A tool that wraps an existing [query engine](/core_modules/query_modules/root.md).

### Blog Post

For full details, please check out our detailed [blog post]().

## LlamaHub
We offer a rich set of Tool Specs that are offered through [LlamaHub](https://llamahub.ai/) 🦙. 
LlamaHub is an open-source repository containing tools and data loaders that you can easily plug and play into any LlamaIndex application.

![](/_static/data_connectors/llamahub.png)

## Usage Pattern

Our Tool Specs and Tools can be imported from the `llama-hub` package.

To use with our agent,
```python
from llama_index.agent import OpenAIAgent
from llama_hub.tools.tool_spec.gmail.base import GmailToolSpec

tool_spec = GmailToolSpec()
agent = OpenAIAgent.from_tools(tool_spec.to_tool_list(), verbose=True)

```

See our Usage Pattern Guide for more details.
```{toctree}
---
maxdepth: 1
---
usage_pattern.md
```

## Tool Example Notebooks

Coming soon! 

