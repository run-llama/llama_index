# Advanced Retrieval Strategies

## Main Advanced Retrieval Strategies

There are a variety of more advanced retrieval strategies you may wish to try, each with different benefits:

- [Reranking](../../examples/node_postprocessor/CohereRerank.ipynb)
- [Recursive retrieval](../../examples/query_engine/pdf_tables/recursive_retriever.ipynb)
- [Embedded tables](../../examples/query_engine/sec_tables/tesla_10q_table.ipynb)
- [Small-to-big retrieval](../../examples/node_postprocessor/MetadataReplacementDemo.ipynb)

See our full [retrievers module guide](../../module_guides/querying/retriever/retrievers.md) for a comprehensive list of all retrieval strategies, broken down into different categories.

- Basic retrieval from each index
- Advanced retrieval and search
- Auto-Retrieval
- Knowledge Graph Retrievers
- Composed/Hierarchical Retrievers
- and more!

More resources are below.

## Query Transformations

A user query can be transformed before it enters a pipeline (query engine, agent, and more). See resources below on query transformations:

- [Query Transform Cookbook](../../examples/query_transformations/query_transform_cookbook.ipynb)
- [Query Transformations Docs](../../optimizing/advanced_retrieval/query_transformations.md)

## Composable Retrievers

Every retriever is capable of retrieving and running other objects, including

- other retrievers
- query engines
- query pipelines
- other nodes

For more details, check out the guide below.

- [Composable Retrievers](../../examples/retrievers/composable_retrievers.ipynb)

## Third-Party Resources

Here are some third-party resources on advanced retrieval strategies.

- [DeepMemory (Activeloop)](../../examples/retrievers/deep_memory.ipynb)
- [Weaviate Hybrid Search](../../examples/vector_stores/WeaviateIndexDemo-Hybrid.ipynb)
- [Pinecone Hybrid Search](../../examples/vector_stores/PineconeIndexDemo-Hybrid.ipynb)
- [Milvus Hybrid Search](../../examples/vector_stores/MilvusHybridIndexDemo.ipynb)
