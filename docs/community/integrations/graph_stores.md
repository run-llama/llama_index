# Using Graph Stores

## `Neo4jGraphStore`

`Neo4j` is supported as a graph store integration. You can persist, visualze, and query graphs using LlamaIndex and Neo4j. Furthermore, existing Neo4j graphs are directly supported using `text2cypher` and the `KnowledgeGraphQueryEngine`.

If you've never used Neo4j before, you can download the desktop client [here](https://neo4j.com/download/).

Once you open the client, create a new project and install the `apoc` integration. Full instructions [here](https://neo4j.com/labs/apoc/4.1/installation/). Just click on your project, select `Plugins` on the left side menu, install APOC and restart your server.

```{toctree}
---
maxdepth: 1
---
Neo4j Graph Store </examples/index_structs/knowledge_graph/Neo4jKGIndexDemo.ipynb>
```

## `NebulaGraphStore`

We support a `NebulaGraphStore` integration, for persisting graphs directly in Nebula! Furthermore, you can generate cypher queries and return natural language responses for your Nebula graphs using the `KnowledgeGraphQueryEngine`.

See the associated guides below:

```{toctree}
---
maxdepth: 1
---
Nebula Graph Store </examples/index_structs/knowledge_graph/NebulaGraphKGIndexDemo.ipynb>
Knowledge Graph Query Engine </examples/query_engine/knowledge_graph_query_engine.ipynb>
```

## `KuzuGraphStore`

We support a `KuzuGraphStore` integration, for persisting graphs directly in [Kuzu](https://kuzudb.com).

See the associated guides below:

```{toctree}
---
maxdepth: 1
---
Kuzu Graph Store </examples/index_structs/knowledge_graph/KuzuGraphDemo.ipynb>
```

## `FalkorDBGraphStore`

We support a `FalkorDBGraphStore` integration, for persisting graphs directly in FalkorDB! Furthermore, you can generate cypher queries and return natural language responses for your FalkorDB graphs using the `KnowledgeGraphQueryEngine`.

See the associated guides below:

```{toctree}
---
maxdepth: 1
---
FalkorDB Graph Store </examples/index_structs/knowledge_graph/FalkorDBGraphKGIndexDemo.ipynb>
```
