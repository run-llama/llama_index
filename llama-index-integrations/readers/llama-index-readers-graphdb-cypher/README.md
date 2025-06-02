# Graph Database Cypher Loader

```bash
pip install llama-index-readers-graphdb-cypher
```

This loader populates documents from results of Cypher queries from a Graph database endpoint.
The user specifies a GraphDB endpoint URL with optional credentials to initialize the reader.
By declaring the Cypher query and optional parameters the loader can fetch the nested result docs.
The results will be turned into a yaml representation to be turned into a string for the document.

The approach should work for Neo4j, AWS Neptune and Memgraph.

## Usage

Here's an example usage of the `GraphDBCypherReader`.

You can test out queries directly with the Neo4j labs demo server: demo.neo4jlabs.com or with a free instance https://neo4j.com/aura

```python
import os

from llama_index.readers.graphdb_cypher import GraphDBCypherReader

uri = "neo4j+s://demo.neo4jlabs.com"
username = "stackoverflow"
password = "stackoverflow"
database = "stackoverflow"

query = """
    MATCH (q:Question)-[:TAGGED]->(:Tag {name:$tag})
    RETURN q.title as title
    ORDER BY q.createdAt DESC LIMIT 10
"""
reader = GraphDBCypherReader(uri, username, password, database)
documents = reader.load_data(query, parameters={"tag": "lua"})
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index)
and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.

It uses the [Neo4j Graph Database](https://neo4j.com/developer) for the Cypher queries.
