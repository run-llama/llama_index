# Module Guides

## Basic

First, check out our [module guide on Indexes](/module_guides/indexing/modules.md) for in-depth guides for each index (vector index, summary index, knowledge graph index). Each index corresponds to a default query engine for that index.

Then check out the rest of the sections below.

```{toctree}
---
maxdepth: 1
---
Custom Query Engine </examples/query_engine/custom_query_engine.ipynb>
Retriever Query Engine </examples/query_engine/CustomRetrievers.ipynb>
```

## Structured & Semi-Structured Data

```{toctree}
---
maxdepth: 1
---
/examples/index_structs/struct_indices/SQLIndexDemo.ipynb
/examples/query_engine/json_query_engine.ipynb
/examples/query_engine/pandas_query_engine.ipynb
/examples/query_engine/knowledge_graph_query_engine.ipynb
/examples/query_engine/knowledge_graph_rag_query_engine.ipynb
/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval.ipynb
```

## Advanced

```{toctree}
---
maxdepth: 1
---
/examples/query_engine/RouterQueryEngine.ipynb
/examples/query_engine/RetrieverRouterQueryEngine.ipynb
/examples/query_engine/JointQASummary.ipynb
/examples/query_engine/sub_question_query_engine.ipynb
/examples/query_transformations/SimpleIndexDemo-multistep.ipynb
/examples/query_engine/SQLRouterQueryEngine.ipynb
/examples/query_engine/SQLAutoVectorQueryEngine.ipynb
/examples/query_engine/SQLJoinQueryEngine.ipynb
/examples/query_engine/pgvector_sql_query_engine.ipynb
/examples/index_structs/struct_indices/duckdb_sql_query.ipynb
Retry Query Engine </examples/evaluation/RetryQuery.ipynb>
/examples/query_engine/citation_query_engine.ipynb
/examples/query_engine/pdf_tables/recursive_retriever.ipynb
/examples/query_engine/sec_tables/tesla_10q_table.ipynb
/examples/query_engine/recursive_retriever_agents.ipynb
/examples/query_engine/ensemble_query_engine.ipynb
```

### Advanced: Towards Multi-Document Querying/Analysis

This specific subsection showcases modules that help with querying multiple documents.

```{toctree}
---
maxdepth: 1
---
/examples/query_engine/sub_question_query_engine.ipynb
/examples/query_engine/recursive_retriever_agents.ipynb
/examples/agent/multi_document_agents.ipynb
/examples/agent/multi_document_agents-v1.ipynb
```

## Experimental

```{toctree}
---
maxdepth: 1
---
/examples/query_engine/flare_query_engine.ipynb
```
