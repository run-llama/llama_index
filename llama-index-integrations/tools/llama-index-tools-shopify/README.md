# Shopify Tool

This tool acts as a custom app for Shopify stores, allowing the Agent to execute GraphQL queries to gather information or perform mutations against the Shopify store.

## Usage

This tool has more extensive example usage documented in a Jupyter notebook [here](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/tools/notebooks/shopify.ipynb)

In particular, the tool is very effective when combined with a method of retrieving data from the GraphQL schema definition.

```bash
pip install llama-index llama-index-readers-file llama-index-tools-shopify unstructured
```

```python
from llama_index.tools.shopify import ShopifyToolSpec
from llama_index.agent.openai import OpenAIAgent

from llama_index.readers.file import UnstructuredReader
from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool

documentation_tool = OnDemandLoaderTool.from_defaults(
    UnstructuredReader(),
    name="graphql_writer",
    description="""
        The GraphQL schema file is located at './data/shopify_graphql.txt', this is always the file argument.
        A tool for processing the Shopify GraphQL spec, and writing queries from the documentation.

        You should pass a query_str to this tool in the form of a request to write a GraphQL query.

        Examples:
            file: './data/shopify_graphql.txt', query_str='Write a graphql query to find unshipped orders'
            file: './data/shopify_graphql.txt', query_str='Write a graphql query to retrieve the stores products'
            file: './data/shopify_graphql.txt', query_str='What fields can you retrieve from the orders object'
        """,
)


shopify_tool = ShopifyToolSpec(
    "your-store.myshopify.com", "2023-04", "your-api-key"
)

agent = OpenAIAgent.from_tools(
    [*shopify_tool.to_tool_list(), documentation_tool],
    system_prompt=f"""
    You are a specialized Agent with access to the Shopify Admin GraphQL API for this Users online store.
    Your job is to chat with store owners and help them run GraphQL queries, interpreting the results for the user

    You can use graphql_writer to query the schema and assist in writing queries.

    If the GraphQL you execute returns an error, either directly fix the query, or directly ask the graphql_writer questions about the schema instead of writing graphql queries.
    Then use that information to write the correct graphql query
    """,
    verbose=True,
    max_function_calls=20,
)

agent.chat("What products are in my store?")
```

`run_graphql_query`: Executes a GraphQL query against the Shopify store

This loader is designed to be used as a way to load data as a Tool in a Agent.
