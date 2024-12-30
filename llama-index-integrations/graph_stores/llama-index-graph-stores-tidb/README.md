# LlamaIndex Graph Stores Integration: TiDB

TiDB is a distributed SQL database, it is MySQL compatible and features horizontal scalability, strong consistency, and high availability. Currently it also supports Vector Search in [TiDB Cloud Serverless](https://tidb.cloud/ai).

In this project, we integrate TiDB as the graph store to store the LlamaIndex graph data, and use TiDB's SQL interface to query the graph data. so that people can use TiDB to interact with LlamaIndex graph index.

- Property Graph Store: `TiDBPropertyGraphStore`
- Knowledge Graph Store: `TiDBGraphStore`

## Installation

```shell
pip install llama-index llama-index-graph-stores-tidb
```

## Usage

### Property Graph Store

NOTE: `TiDBPropertyGraphStore` requires the Vector Search feature in TiDB, but now it is only available in [TiDB Cloud Serverless](https://tidb.cloud/ai).

Please checkout this [tutorial](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/property_graph/property_graph_tidb.ipynb) to learn how to use `TiDBPropertyGraphStore` with LlamaIndex.

Simple example to use `TiDBPropertyGraphStore`:

```python
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.graph_stores.tidb import TiDBPropertyGraphStore

documents = SimpleDirectoryReader(
    "../../../examples/data/paul_graham/"
).load_data()

graph_store = TiDBPropertyGraphStore(
    db_connection_string="mysql+pymysql://user:password@host:4000/dbname?ssl_verify_cert=true&ssl_verify_identity=true",
)

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
    kg_extractors=[
        SchemaLLMPathExtractor(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.0)
        )
    ],
    property_graph_store=graph_store,
    show_progress=True,
)

query_engine = index.as_query_engine(include_text=True)
response = query_engine.query("What happened at Interleaf and Viaweb?")
print(response)
```

### Knowledge Graph Store

Checkout this [tutorial](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/index_structs/knowledge_graph/TiDBKGIndexDemo.ipynb) to learn how to use `TiDBGraphStore` with LlamaIndex.

For `TiDBGraphStore`, you can use either Self-Hosted TiDB or TiDB Cloud Serverless(Recommended).

Simple example to use `TiDBGraphStore`:

```python
from llama_index.graph_stores.tidb import TiDBGraphStore
from llama_index.core import (
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
)

documents = SimpleDirectoryReader(
    "../../../examples/data/paul_graham/"
).load_data()

graph_store = TiDBGraphStore(
    db_connection_string="mysql+pymysql://user:password@host:4000/dbname"
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)
index = KnowledgeGraphIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
)
query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about Interleaf",
)
print(response)
```
