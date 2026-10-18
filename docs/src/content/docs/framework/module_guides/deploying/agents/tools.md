---
title: Tools
---

## Concept

Having proper tool abstractions is at the core of building [agentic systems in LlamaIndex](/python/framework/module_guides/deploying/agents). Defining a set of Tools is similar to defining any API interface, with the exception that these Tools are meant for agent rather than human use. We allow users to define both a **Tool** as well as a **ToolSpec** containing a series of functions under the hood.

When using an agent or LLM with function calling, the tool selected (and the arguments written for that tool) rely strongly on the **tool name** and **description** of the tools purpose and arguments. Spending time tuning these parameters can result in larges changes in how the LLM calls these tools.

A Tool implements a very generic interface - simply define `__call__` and also return some basic metadata (name, description, function schema).

We offer a few different types of Tools:

- `FunctionTool`: A function tool allows users to easily convert any user-defined function into a Tool. It can also auto-infer the function schema, or let you customize various aspects.
- `QueryEngineTool`: A tool that wraps an existing [query engine](/python/framework/module_guides/deploying/query_engine). Note: since our agent abstractions inherit from `BaseQueryEngine`, these tools can also wrap other agents.
- Community contributed `ToolSpecs` that define one or more tools around a single service (like Gmail)
- Utility tools for wrapping other tools to handle returning large amounts of data from a tool

## FunctionTool

A function tool is a simple wrapper around any existing function (both sync and async are supported!).

```python
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool


def get_weather(location: str) -> str:
    """Usfeful for getting the weather for a given location."""
    ...


tool = FunctionTool.from_defaults(
    get_weather,
    # async_fn=aget_weather,  # optional!
)

agent = ReActAgent(llm=llm, tools=tools)
```

For a better function definition, you can also leverage the `Annotated` type to specify argument descriptions.

```python
from typing import Annotated


def get_weather(
    location: Annotated[
        str, "A city name and state, formatted like '<name>, <state>'"
    ],
) -> str:
    """Useful for getting the weather for a given location."""
    ...


tool = FunctionTool.from_defaults(get_weather)
```

By default, the tool name will be the function name, and the docstring will be the tool description. But you can also override this.

```python
tool = FunctionTool.from_defaults(get_weather, name="...", description="...")
```

## QueryEngineTool

Any query engine can be turned into a tool, using `QueryEngineTool`:

```python
from llama_index.core.tools import QueryEngineTool

tool = QueryEngineTool.from_defaults(
    query_engine, name="...", description="..."
)
```

## Tool Specs

We also offer a rich set of Tools and Tool Specs as [separate integration packages](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/tools).

You can think of tool specs like bundles of tools meant to be used together. Usually these cover useful tools across a single interface/service, like Gmail.

To use with an agent, you can install the specific tool spec integration:

```bash
pip install llama-index-tools-google
```

And then use it:

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.google import GmailToolSpec

tool_spec = GmailToolSpec()
agent = FunctionAgent(llm=llm, tools=tool_spec.to_tool_list())
```

See the [tools directory on GitHub](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/tools) for a full list of community contributed tool specs.

### Externally maintained tool specs

New integration packages can be maintained in independent repositories, as described
in the [contribution guide](https://github.com/run-llama/llama_index/blob/main/CONTRIBUTING.md).
They use the same `BaseToolSpec.to_tool_list()` interface. Check the maintainer,
installation source, credential requirements, and release status before use.

- [Suno by Ace Data Cloud](https://github.com/AceDataCloud/SunoLlamaIndex): native
  music generation and retrieval of the same asynchronous task. The independent
  package is `llama-index-tools-suno-acedatacloud`, with
  `from llama_index.tools.suno_acedatacloud import SunoToolSpec`. Its [v0.1.0 release](https://github.com/AceDataCloud/SunoLlamaIndex/releases/tag/v0.1.0)
  provides an installable wheel and source archive. This integration is
  maintained by Ace Data Cloud; it is not an official Suno product. Generation may
  consume account credits. A pending audio preview is not a completed result.

- [Seedance by Ace Data Cloud](https://github.com/AceDataCloud/SeedanceLlamaIndex):
  native text-to-video, first-frame animation, and retrieval of the same task;
  package `llama-index-tools-seedance-acedatacloud`.
- [HappyHorse by Ace Data Cloud](https://github.com/AceDataCloud/HappyHorseLlamaIndex):
  text-to-video and first-frame animation with explicit action-matched models;
  package `llama-index-tools-happyhorse-acedatacloud`.
- [GPT Image by Ace Data Cloud](https://github.com/AceDataCloud/GPTImageLlamaIndex):
  image generation/editing and same-task retrieval; completed image tasks require
  a `finished_at` timestamp. Package `llama-index-tools-gpt-image-acedatacloud`.
- [Nano Banana by Ace Data Cloud](https://github.com/AceDataCloud/NanoBananaLlamaIndex):
  image generation/editing and same-task retrieval;
  package `llama-index-tools-nano-banana-acedatacloud`.

These independently maintained packages provide synchronous and asynchronous native
FunctionTools. Version 0.1.0 wheels and source archives are published on each linked repository’s
GitHub Releases page; the README gives the exact installation URL. PyPI publication
is pending registry project-creation limits, so a package-name install is not yet
claimed. These entries do not imply default inclusion or endorsement. Each requires an
Ace application token. Generation may consume credits; save the returned task ID
and query it rather than submitting again while pending.

## Utility Tools

Oftentimes, directly querying an API can return a massive volume of data, which on its own may overflow the context window of the LLM (or at the very least unnecessarily increase the number of tokens that you are using).

To tackle this, we’ve provided an initial set of “utility tools” in our tool integrations - utility tools are not conceptually tied to a given service (e.g. Gmail, Notion), but rather can augment the capabilities of existing Tools. In this particular case, utility tools help to abstract away common patterns of needing to cache/index and query data that’s returned from any API request.

Let’s walk through our two main utility tools below.

### OnDemandLoaderTool

This tool turns any existing LlamaIndex data loader ( `BaseReader` class) into a tool that an agent can use. The tool can be called with all the parameters needed to trigger `load_data` from the data loader, along with a natural language query string. During execution, we first load data from the data loader, index it (for instance with a vector store), and then query it “on-demand”. All three of these steps happen in a single tool call.

Oftentimes this can be preferable to figuring out how to load and index API data yourself. While this may allow for data reusability, oftentimes users just need an ad-hoc index to abstract away prompt window limitations for any API call.

A usage example is given below:

```bash
pip install llama-index-readers-wikipedia
```

```python
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool

tool = OnDemandLoaderTool.from_defaults(
    reader,
    name="Wikipedia Tool",
    description="A tool for loading data and querying articles from Wikipedia",
)
```

### LoadAndSearchToolSpec

The LoadAndSearchToolSpec takes in any existing Tool as input. As a tool spec, it implements `to_tool_list` , and when that function is called, two tools are returned: a `load` tool and then a `search` tool.

The `load` Tool execution would call the underlying Tool, and the index the output (by default with a vector index). The `search` Tool execution would take in a query string as input and call the underlying index.

This is helpful for any API endpoint that will by default return large volumes of data - for instance our WikipediaToolSpec will by default return entire Wikipedia pages, which will easily overflow most LLM context windows.

Example usage is shown below:

```bash
pip install llama-index-tools-wikipedia
```

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools.tool_spec.load_and_search import (
    LoadAndSearchToolSpec,
)
from llama_index.tools.wikipedia import WikipediaToolSpec


wiki_spec = WikipediaToolSpec()
# Get the search wikipedia tool
tool = wiki_spec.to_tool_list()[1]

# Create the Agent with load/search tools
agent = FunctionAgent(
    llm=llm, tools=LoadAndSearchToolSpec.from_defaults(tool).to_tool_list()
)
```

### Return Direct

You'll notice the option `return_direct` in the tool class constructor. If this is set to `True`, the response from an agent is returned directly, without being interpreted and rewritten by the agent. This can be helpful for decreasing runtime, or designing/specifying tools that will end the agent reasoning loop.

For example, say you specify a tool:

```python
tool = QueryEngineTool.from_defaults(
    query_engine,
    name="<name>",
    description="<description>",
    return_direct=True,
)

agent = FunctionAgent(llm=llm, tools=[tool])

response = await agent.run("<question that invokes tool>")
```

In the above example, the query engine tool would be invoked, and the response from that tool would be directly returned as the response, and the execution loop would end.

If `return_direct=False` was used, then the agent would rewrite the response using the context of the chat history, or even make another tool call.

We have also provided an [example notebook](/python/examples/agent/return_direct_agent) of using `return_direct`.

## Debugging Tools

Often, it can be useful to debug what exactly the tool definition is that is being sent to APIs.

You can get a good peek at this by using the underlying function to get the current tool schema, which is levereged in APIs like OpenAI and Anthropic.

```python
schema = tool.metadata.get_parameters_dict()
print(schema)
```
