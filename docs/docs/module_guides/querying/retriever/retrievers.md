# Retriever Modules

We are actively adding more tailored retrieval guides.
In the meanwhile, please take a look at the [API References](../../../api_reference/retrievers/index.md).

## Index Retrievers

Please see [the retriever modes](retriever_modes.md) for more details on how to get a retriever from any given index.

If you want to import the corresponding retrievers directly, please check out our [API reference](../../../api_reference/retrievers/index.md).

## Comprehensive Retriever Guides

Check out our comprehensive guides on various retriever modules, many of which cover advanced concepts (auto-retrieval, routing, ensembling, and more).

### Advanced Retrieval and Search

These guides contain advanced retrieval techniques. Some are common like keyword/hybrid search, reranking, and more.
Some are specific to LLM + RAG pipelines, like small-to-big and auto-merging retrieval.

- [Define Custom Retriever](../../../examples/query_engine/CustomRetrievers.ipynb)
- [BM25 Hybrid Retriever](../../../examples/retrievers/bm25_retriever.ipynb)
- [Simple Query Fusion](../../../examples/retrievers/simple_fusion.ipynb)
- [Reciprocal Rerank Fusion](../../../examples/retrievers/reciprocal_rerank_fusion.ipynb)
- [Auto Merging Retriever](../../../examples/retrievers/auto_merging_retriever.ipynb)
- [Metadata Replacement](../../../examples/node_postprocessor/MetadataReplacementDemo.ipynb)
- [Composable Retrievers](../../../examples/retrievers/composable_retrievers.ipynb)

### Auto-Retrieval

These retrieval techniques perform **semi-structured** queries, combining semantic search with structured filtering.

- [Auto-Retrieval (with Pinecone)](../../../examples/vector_stores/pinecone_auto_retriever.ipynb)
- [Auto-Retrieval (with Lantern)](../../../examples/vector_stores/LanternAutoRetriever.ipynb)
- [Auto-Retrieval (with Chroma)](../../../examples/vector_stores/chroma_auto_retriever.ipynb)
- [Auto-Retrieval (with BagelDB)](../../../examples/vector_stores/BagelAutoRetriever.ipynb)
- [Auto-Retrieval (with Vectara)](../../../examples/retrievers/vectara_auto_retriever.ipynb)
- [Multi-Doc Auto-Retrieval](../../../examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval.ipynb)

### Knowledge Graph Retrievers

- [Knowledge Graph RAG Retriever](../../../examples/query_engine/knowledge_graph_rag_query_engine.ipynb)

### Composed Retrievers

These are retrieval techniques that are composed on top of other retrieval techniques - providing higher-level capabilities like
hierarchical retrieval and query decomposition.

- [Query Fusion](../../../examples/retrievers/reciprocal_rerank_fusion.ipynb)
- [Recursive Table Retrieval](../../../examples/query_engine/pdf_tables/recursive_retriever.ipynb)
- [Recursive Node Retrieval](../../../examples/retrievers/recursive_retriever_nodes.ipynb)
- [Braintrust](../../../examples/retrievers/recurisve_retriever_nodes_braintrust.ipynb)
- [Router Retriever](../../../examples/retrievers/router_retriever.ipynb)
- [Ensemble Retriever](../../../examples/retrievers/ensemble_retrieval.ipynb)
- [Multi-Doc Auto-Retrieval](../../../examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval.ipynb)

### Managed Retrievers

- [Google](../../../examples/managed/GoogleDemo.ipynb)
- [Vectara](../../../examples/managed/vectaraDemo.ipynb)
- [VideoDB](../../../examples/retrievers/videodb_retriever.ipynb)
- [Zilliz](../../../examples/managed/zcpDemo.ipynb)
- [Amazon Bedrock](../../../examples/retrievers/bedrock_retriever.ipynb)

### Other Retrievers

These are guides that don't fit neatly into a category but should be highlighted regardless.

- [Multi-Doc Hybrid](../../../examples/retrievers/multi_doc_together_hybrid.ipynb)
- [You Retriever](../../../examples/retrievers/you_retriever.ipynb)
- [Text-to-SQL](../../../examples/index_structs/struct_indices/SQLIndexDemo.ipynb)
- [DeepMemory (Activeloop)](../../../examples/retrievers/deep_memory.ipynb)
- [Pathway](../../../examples/retrievers/pathway_retriever.ipynb)
