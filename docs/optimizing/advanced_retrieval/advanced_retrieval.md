# Advanced Retrieval Strategies

## Main Advanced Retrieval Strategies

There are a variety of more advanced retrieval strategies you may wish to try, each with different benefits:

- {ref}`Reranking <cohere_rerank>`
- [Recursive retrieval](/examples/query_engine/pdf_tables/recursive_retriever.ipynb)
- [Embedded tables](/examples/query_engine/sec_tables/tesla_10q_table.ipynb)
- [Small-to-big retrieval](/examples/node_postprocessor/MetadataReplacementDemo.ipynb)

See our full [retrievers module guide](/module_guides/querying/retriever/retrievers.md) for a comprehensive list of all retrieval strategies, broken down into different categories.

- Basic retrieval from each index
- Advanced retrieval and search
- Auto-Retrieval
- Knowledge Graph Retrievers
- Composed/Hierarchical Retrievers
- and more!

More resources are below.

## Query Transformations

A user query can be transformed before it enters a pipeline (query engine, agent, and more). See resources below on query transformations:

```{toctree}
---
maxdepth: 1
---
/examples/query_transformations/query_transform_cookbook.ipynb
/optimizing/advanced_retrieval/query_transformations.md
```

## Third-Party Resources

Here are some third-party resources on advanced retrieval strategies.

```{toctree}
---
maxdepth: 1
---
DeepMemory (Activeloop) </examples/retrievers/deep_memory.ipynb>
/examples/vector_stores/WeaviateIndexDemo-Hybrid.ipynb
/examples/vector_stores/PineconeIndexDemo-Hybrid.ipynb
```

## Structured Outputs

You may want to ensure your output is structured. See our comprehensive guides below to see how to do that.

```{toctree}
---
maxdepth: 1
---
/optimizing/advanced_retrieval/structured_outputs/structured_outputs.md
/optimizing/advanced_retrieval/structured_outputs/pydantic_program.md
/optimizing/advanced_retrieval/structured_outputs/query_engine.md
```
