# VectorDB Tool

This tool wraps a VectorStoreIndex and enables a agent to call it with queries and filters to retrieve data.

## Usage

```python
from llama_index.tools.vector_db import VectorDB
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.vector_stores import VectorStoreInfo
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex(nodes=nodes)
tool_spec = VectorDB(index=index)
vector_store_info = VectorStoreInfo(
    content_info="brief biography of celebrities",
    metadata_info=[
        MetadataInfo(
            name="category",
            type="str",
            description="Category of the celebrity, one of [Sports, Entertainment, Business, Music]",
        ),
        MetadataInfo(
            name="country",
            type="str",
            description="Country of the celebrity, one of [United States, Barbados, Portugal]",
        ),
    ],
)

agent = OpenAIAgent.from_tools(
    tool_spec.to_tool_list(
        func_to_metadata_mapping={
            "auto_retrieve_fn": ToolMetadata(
                name="celebrity_bios",
                description=f"""\
            Use this tool to look up biographical information about celebrities.
            The vector database schema is given below:

            {vector_store_info.json()}

            {tool_spec.auto_retrieve_fn.__doc__}
        """,
                fn_schema=create_schema_from_function(
                    "celebrity_bios", tool_spec.auto_retrieve_fn
                ),
            )
        }
    ),
    verbose=True,
)

agent.chat("Tell me about two celebrities from the United States. ")
```

`auto_retrieve_fn`: Retrieves data from the index

This loader is designed to be used as a way to load data as a Tool in a Agent.
