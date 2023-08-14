# Module Guides
We are adding more module guides soon!
In the meanwhile, please take a look at the [API References](/api_reference/query/retrievers.rst).

## Vector Index Retrievers
* VectorIndexRetriever
```{toctree}
---
maxdepth: 1
---
VectorIndexAutoRetriever </examples/vector_stores/chroma_auto_retriever.ipynb>
```

## List Index
* ListIndexRetriever 
* ListIndexEmbeddingRetriever 
* ListIndexLLMRetriever

## Tree Index
* TreeSelectLeafRetriever
* TreeSelectLeafEmbeddingRetriever
* TreeAllLeafRetriever
* TreeRootRetriever


## Keyword Table Index
* KeywordTableGPTRetriever
* KeywordTableSimpleRetriever
* KeywordTableRAKERetriever


## Knowledge Graph Index
```{toctree}
---
maxdepth: 1
---
Custom Retriever (KG Index and Vector Store Index) </examples/index_structs/knowledge_graph/KnowledgeGraphIndex_vs_VectorStoreIndex_vs_CustomIndex_combined.ipynb>
```
* KGTableRetriever

## Document Summary Index
* DocumentSummaryIndexRetriever
* DocumentSummaryIndexEmbeddingRetriever

## Composed Retrievers
* TransformRetriever
```{toctree}
---
maxdepth: 1
---
/examples/query_engine/pdf_tables/recursive_retriever.ipynb
/examples/retrievers/router_retriever.ipynb
/examples/retrievers/ensemble_retrieval.ipynb
```
