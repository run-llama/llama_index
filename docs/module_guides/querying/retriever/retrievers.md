# Retriever Modules

We are actively adding more tailored retrieval guides.
In the meanwhile, please take a look at the [API References](/api_reference/query/retrievers.rst).

## Index Retrievers

Please see [the retriever modes](/module_guides/querying/retriever/retriever_modes.md) for more details on how to get a retriever from any given index.

If you want to import the corresponding retrievers directly, please check out our [API reference](/api_reference/query/retrievers.rst).

## Comprehensive Retriever Guides

Check out our comprehensive guides on various retriever modules, many of which cover advanced concepts (auto-retrieval, routing, ensembling, and more).

### Advanced Retrieval and Search

These guides contain advanced retrieval techniques. Some are common like keyword/hybrid search, reranking, and more.
Some are specific to LLM + RAG pipelines, like small-to-big and auto-merging retrieval.

```{toctree}
---
maxdepth: 1
---
Define Custom Retriever </examples/query_engine/CustomRetrievers.ipynb>
BM25 Hybrid Retriever </examples/retrievers/bm25_retriever.ipynb>
/examples/retrievers/simple_fusion.ipynb
/examples/retrievers/reciprocal_rerank_fusion.ipynb
/examples/retrievers/auto_merging_retriever.ipynb
/examples/node_postprocessor/MetadataReplacementDemo.ipynb
```

### Auto-Retrieval

These retrieval techniques perform **semi-structured** queries, combining semantic search with structured filtering.

```{toctree}
---
maxdepth: 1
---
/examples/vector_stores/pinecone_auto_retriever.ipynb
Auto-Retrieval (with Chroma) </examples/vector_stores/chroma_auto_retriever.ipynb>
Auto-Retrieval (with BagelDB) </examples/vector_stores/BagelAutoRetriever.ipynb>
/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval.ipynb
/examples/retrievers/vectara_auto_retriever.ipynb
```

### Knowledge Graph Retrievers

```{toctree}
---
maxdepth: 1
---
Custom Retriever (KG Index and Vector Store Index) </examples/index_structs/knowledge_graph/KnowledgeGraphIndex_vs_VectorStoreIndex_vs_CustomIndex_combined.ipynb>
Knowledge Graph RAG Retriever </examples/query_engine/knowledge_graph_rag_query_engine.ipynb>
```

### Composed Retrievers

These are retrieval techniques that are composed on top of other retrieval techniques - providing higher-level capabilities like
hierarchical retrieval and query decomposition.

```{toctree}
---
maxdepth: 1
---
/examples/query_engine/pdf_tables/recursive_retriever.ipynb
/examples/retrievers/recursive_retriever_nodes.ipynb
/examples/retrievers/recurisve_retriever_nodes_braintrust.ipynb
/examples/retrievers/router_retriever.ipynb
/examples/retrievers/ensemble_retrieval.ipynb
/examples/managed/GoogleDemo.ipynb
/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval.ipynb
```

### Managed Retrievers

```{toctree}
---
maxdepth: 1
---
/examples/managed/GoogleDemo.ipynb
/examples/managed/vectaraDemo.ipynb
/examples/managed/zcpDemo.ipynb
```

### Other Retrievers

These are guides that don't fit neatly into a category but should be highlighted regardless.

```{toctree}
---
maxdepth: 1
---
/examples/retrievers/you_retriever.ipynb
/examples/index_structs/struct_indices/SQLIndexDemo.ipynb
DeepMemory (Activeloop) </examples/retrievers/deep_memory.ipynb>
/examples/retrievers/pathway_retriever.ipynb
```
