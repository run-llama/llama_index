---
title: Advanced Retrieval Strategies
---

## Main Advanced Retrieval Strategies

There are a variety of more advanced retrieval strategies you may wish to try, each with different benefits:

- [Reranking](/python/examples/node_postprocessor/coherererank)
- [Recursive retrieval](/python/examples/query_engine/pdf_tables/recursive_retriever)
- [Embedded tables](/python/examples/query_engine/sec_tables/tesla_10q_table)
- [Small-to-big retrieval](/python/examples/node_postprocessor/metadatareplacementdemo)

See our full [retrievers module guide](/python/framework/module_guides/querying/retriever/retrievers) for a comprehensive list of all retrieval strategies, broken down into different categories.

- Basic retrieval from each index
- Advanced retrieval and search
- Auto-Retrieval
- Knowledge Graph Retrievers
- Composed/Hierarchical Retrievers
- and more!

More resources are below.

## Query Transformations

A user query can be transformed before it enters a flow (query engine, agent, and more). See resources below on query transformations:

- [Query Transform Cookbook](/python/examples/query_transformations/query_transform_cookbook)
- [Query Transformations Docs](/python/framework/optimizing/advanced_retrieval/query_transformations)

## Composable Retrievers

Every retriever is capable of retrieving and running other objects, including

- other retrievers
- query engines
- query pipelines
- other nodes

For more details, check out the guide below.

- [Composable Retrievers](/python/examples/retrievers/composable_retrievers)

## Third-Party Resources

Here are some third-party resources on advanced retrieval strategies.

- [DeepMemory (Activeloop)](/python/examples/retrievers/deep_memory)
- [Weaviate Hybrid Search](/python/examples/vector_stores/weaviateindexdemo-hybrid)
- [Pinecone Hybrid Search](/python/examples/vector_stores/pineconeindexdemo-hybrid)
- [Milvus Hybrid Search](/python/examples/vector_stores/milvushybridindexdemo)
