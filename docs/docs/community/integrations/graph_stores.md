# Using Graph Stores

## `Neo4jGraphStore`

`Neo4j` is supported as a graph store integration. You can persist, visualize, and query graphs using LlamaIndex and Neo4j. Furthermore, existing Neo4j graphs are directly supported using `text2cypher` and the `KnowledgeGraphQueryEngine`.

If you've never used Neo4j before, you can download the desktop client [here](https://neo4j.com/download/).

Once you open the client, create a new project and install the `apoc` integration. Full instructions [here](https://neo4j.com/labs/apoc/4.1/installation/). Just click on your project, select `Plugins` on the left side menu, install APOC and restart your server.

See the example of using the [Neo4j Graph Store](/python/examples/index_structs/knowledge_graph/neo4jkgindexdemo).

## `NebulaGraphStore`

We support a `NebulaGraphStore` integration, for persisting graphs directly in Nebula! Furthermore, you can generate cypher queries and return natural language responses for your Nebula graphs using the `KnowledgeGraphQueryEngine`.

See the associated guides below:

- [Nebula Graph Store](/python/examples/index_structs/knowledge_graph/nebulagraphkgindexdemo)
- [Knowledge Graph Query Engine](/python/examples/query_engine/knowledge_graph_query_engine)

## `KuzuGraphStore`

We support a `KuzuGraphStore` integration, for persisting triples directly in [Kuzu](https://kuzudb.com).
Additionally, we support the `PropertyGraphIndex`, which allows you to store and query property graphs
using a Kuzu backend.

See the associated guides below:

- [Kuzu Graph Store](/python/examples/index_structs/knowledge_graph/kuzugraphdemo)
- [Kuzu Graph Store](/python/examples/property_graph/property_graph_kuzu)

## `FalkorDBGraphStore`

We support a `FalkorDBGraphStore` integration, for persisting graphs directly in FalkorDB! Furthermore, you can generate cypher queries and return natural language responses for your FalkorDB graphs using the `KnowledgeGraphQueryEngine`.

See the associated guides below:

- [FalkorDB Graph Store](/python/examples/index_structs/knowledge_graph/falkordbgraphdemo)

## `Amazon Neptune Graph Stores`

We support `Amazon Neptune` integrations for both [Neptune Database](https://docs.aws.amazon.com/neptune/latest/userguide/feature-overview.html) and [Neptune Analytics](https://docs.aws.amazon.com/neptune-analytics/latest/userguide/what-is-neptune-analytics.html) as a graph store integration.

See the associated guides below:

- [Amazon Neptune Graph Store](/python/examples/index_structs/knowledge_graph/neptunedatabasekgindexdemo).


## `TiDB Graph Store`

We support a `TiDBGraphStore` integration, for persisting graphs directly in [TiDB](https://docs.pingcap.com/tidb/stable/overview)!

See the associated guides below:

- [TiDB Graph Store](/python/examples/index_structs/knowledge_graph/tidbkgindexdemo)
