# GraphQL Tool

This tool provides agents the ability to easily execute GraphQL queries against a server. The tool can be initialized with the server url and any required headers and thereafter perform queries against the server

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-graphql/examples/graphql.ipynb)

Here's an example usage of the GraphQLToolSpec.

This tool works best when the Agent has access to the GraphQL schema for the server. See [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-shopify/examples/shopify.ipynb) for an example of using a tool with a file loader to create even more powerful Agents.

```python
from llama_index.tools.graphql import GraphQLToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = GraphQLToolSpec(
    url="https://spacex-production.up.railway.app/",
    headers={
        "content-type": "application/json",
    },
)

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

await agent.run(
    "get the id, model, name and type of the Ships from the graphql endpoint"
)
```

`graphql_request`: Runs a GraphQL query against the configured server

This loader is designed to be used as a way to load data as a Tool in a Agent.
