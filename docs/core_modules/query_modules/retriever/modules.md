# Module Guides
We are actively adding more tailored retrieval guides.
In the meanwhile, please take a look at the [API References](/api_reference/query/retrievers.rst).

## Index Retrievers

Please see [the retriever modes](/core_modules/query_modules/retriever/retriever_modes.md) for more details on how to get a retriever from any given index.

If you want to import the corresponding retrievers directly, please check out our [API reference](/api_reference/query/retrievers.rst).

## Advanced Retriever Guides

Check out our comprehensive guides on various retriever modules, many of which cover advanced concepts (auto-retrieval, routing, ensembling, and more).

## External Retrievers
```{toctree}
---
maxdepth: 1
---
/examples/retrievers/bm25_retriever.ipynb 
```

## Knowledge Graph Retrievers
```{toctree}
---
maxdepth: 1
---
Custom Retriever (KG Index and Vector Store Index) </examples/index_structs/knowledge_graph/KnowledgeGraphIndex_vs_VectorStoreIndex_vs_CustomIndex_combined.ipynb>
Knowledge Graph RAG Retriever </examples/query_engine/knowledge_graph_rag_query_engine.ipynb>
```

## Composed Retrievers
```{toctree}
---
maxdepth: 1
---
Auto-Retrieval (with Chroma) </examples/vector_stores/chroma_auto_retriever.ipynb>
Auto-Retrieval (with BagelDB) </examples/vector_stores/BagelAutoRetriever.ipynb>
/examples/query_engine/pdf_tables/recursive_retriever.ipynb
/examples/retrievers/router_retriever.ipynb
/examples/retrievers/ensemble_retrieval.ipynb
/examples/retrievers/auto_merging_retriever.ipynb
/examples/retrievers/recursive_retriever_nodes.ipynb
```
