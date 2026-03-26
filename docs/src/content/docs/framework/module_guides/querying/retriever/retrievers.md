---
title: Retriever Modules
---

We are actively adding more tailored retrieval guides.
In the meanwhile, please take a look at the [API References](/python/framework-api-reference/retrievers).

## Index Retrievers

Please see [the retriever modes](/python/framework/module_guides/querying/retriever/retriever_modes) for more details on how to get a retriever from any given index.

If you want to import the corresponding retrievers directly, please check out our [API reference](/python/framework-api-reference/retrievers).

## Comprehensive Retriever Guides

Check out our comprehensive guides on various retriever modules, many of which cover advanced concepts (auto-retrieval, routing, ensembling, and more).

### Advanced Retrieval and Search

These guides contain advanced retrieval techniques. Some are common like keyword/hybrid search, reranking, and more.
Some are specific to LLM + RAG workflows, like small-to-big and auto-merging retrieval.

- [Define Custom Retriever](/python/examples/query_engine/customretrievers)
- [BM25 Hybrid Retriever](/python/examples/retrievers/bm25_retriever)
- [Simple Query Fusion](/python/examples/retrievers/simple_fusion)
- [Reciprocal Rerank Fusion](/python/examples/retrievers/reciprocal_rerank_fusion)
- [Auto Merging Retriever](/python/examples/retrievers/auto_merging_retriever)
- [Metadata Replacement](/python/examples/node_postprocessor/metadatareplacementdemo)
- [Composable Retrievers](/python/examples/retrievers/composable_retrievers)

### Auto-Retrieval

These retrieval techniques perform **semi-structured** queries, combining semantic search with structured filtering.

- [Auto-Retrieval (with Pinecone)](/python/examples/vector_stores/pinecone_auto_retriever)
- [Auto-Retrieval (with Lantern)](/python/examples/vector_stores/lanternautoretriever)
- [Auto-Retrieval (with Chroma)](/python/examples/vector_stores/chroma_auto_retriever)
- [Auto-Retrieval (with BagelDB)](/python/examples/vector_stores/bagelautoretriever)
- [Auto-Retrieval (with Vectara)](/python/examples/retrievers/vectara_auto_retriever)
- [Multi-Doc Auto-Retrieval](/python/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval)

### Knowledge Graph Retrievers

- [Knowledge Graph RAG Retriever](/python/examples/query_engine/knowledge_graph_rag_query_engine)

### Composed Retrievers

These are retrieval techniques that are composed on top of other retrieval techniques - providing higher-level capabilities like
hierarchical retrieval and query decomposition.

- [Query Fusion](/python/examples/retrievers/reciprocal_rerank_fusion)
- [Recursive Table Retrieval](/python/examples/query_engine/pdf_tables/recursive_retriever)
- [Recursive Node Retrieval](/python/examples/retrievers/recursive_retriever_nodes)
- [Braintrust](/python/examples/retrievers/recurisve_retriever_nodes_braintrust)
- [Router Retriever](/python/examples/retrievers/router_retriever)
- [Ensemble Retriever](/python/examples/retrievers/ensemble_retrieval)
- [Multi-Doc Auto-Retrieval](/python/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval)

### Managed Retrievers

- [Google](/python/examples/managed/googledemo)
- [Vectara](/python/examples/managed/vectarademo)
- [VideoDB](/python/examples/retrievers/videodb_retriever)
- [Amazon Bedrock](/python/examples/retrievers/bedrock_retriever)

### Other Retrievers

These are guides that don't fit neatly into a category but should be highlighted regardless.

- [Multi-Doc Hybrid](/python/examples/retrievers/multi_doc_together_hybrid)
- [You Retriever](/python/examples/retrievers/you_retriever)
- [Text-to-SQL](/python/examples/index_structs/struct_indices/sqlindexdemo)
- [DeepMemory (Activeloop)](/python/examples/retrievers/deep_memory)
- [Pathway](/python/examples/retrievers/pathway_retriever)
