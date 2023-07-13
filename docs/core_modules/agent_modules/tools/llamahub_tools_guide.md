# LlamaHub Tools Guide

We offer a rich set of Tool Specs that are offered through [LlamaHub](https://llamahub.ai/) 🦙. 
![](/_static/data_connectors/llamahub.png)

These tool specs represent an initial curated list of services that an agent can interact with and enrich its capability to perform different actions. 

We also provide a list of **utility tools** that help to abstract away pain points when designing agents to interact with different API services that return large amounts of data.

## Tool Specs

Coming soon! 

## Utility Tools

Oftentimes, directly querying an API can return a massive volume of data, which on its own may overflow the context window of the LLM (or at the very least unnecessarily increase the number of tokens that you are using). 

To tackle this, we’ve provided an initial set of “utility tools” in LlamaHub Tools - utility tools are not conceptually tied to a given service (e.g. Gmail, Notion), but rather can augment the capabilities of existing Tools. In this particular case, utility tools help to abstract away common patterns of needing to cache/index and query data that’s returned from any API request.

Let’s walk through our two main utility tools below.

### OnDemandLoaderTool

This tool turns any existing LlamaIndex data loader ( `BaseReader` class) into a tool that an agent can use. The tool can be called with all the parameters needed to trigger `load_data` from the data loader, along with a natural language query string. During execution, we first load data from the data loader, index it (for instance with a vector store), and then query it “on-demand”. All three of these steps happen in a single tool call.

Oftentimes this can be preferable to figuring out how to load and index API data yourself. While this may allow for data reusability, oftentimes users just need an ad-hoc index to abstract away prompt window limitations for any API call. 

A usage example is given below:

```python
from llama_hub.wikipedia.base import WikipediaReader
from llama_index.tools.on_demand_loader_tool import OnDemandLoaderTool

tool = OnDemandLoaderTool.from_defaults(
	reader,
	name="Wikipedia Tool",
	description="A tool for loading data and querying articles from Wikipedia"
)
```

### LoadAndSearchToolSpec

The LoadAndSearchToolSpec takes in any existing Tool as input. As a tool spec, it implements `to_tool_list` , and when that function is called, two tools are returned: a `load` tool and then a `search` tool.

The `load` Tool execution would call the underlying Tool, and the index the output (by default with a vector index). The `search` Tool execution would take in a query string as input and call the underlying index.

This is helpful for any API endpoint that will by default return large volumes of data - for instance our WikipediaToolSpec will by default return entire Wikipedia pages, which will easily overflow most LLM context windows.

Example usage is shown below:

```python
from llama_hub.tools.wikipedia.base import WikipediaToolSpec
from llama_index.tools.load_and_search.base import LoadAndSearchToolSpec

wiki_spec = WikipediaToolSpec()
# Get the search wikipedia tool
tool = wiki_spec.to_tool_list()[1]

# Create the Agent with load/search tools
agent = OpenAIAgent.from_tools(
 LoadAndSearchToolSpec.from_defaults(
    tool
 ).to_tool_list(), verbose=True
)
```
