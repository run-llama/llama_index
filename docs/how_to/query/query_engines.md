# Query Engines

Query engine is a generic interface that takes in a query and returns a response.
Query engines can be implemented by composing retrievers, response synthesizer modules.
They can also be built on top of other query engines.  



```{toctree}
---
caption: Examples
maxdepth: 1
---
../../examples/query_engine/CustomRetrievers.ipynb
../../examples/query_engine/RouterQueryEngine.ipynb
../../examples/query_engine/RetrieverRouterQueryEngine.ipynb
../../examples/query_engine/JointQASummary.ipynb
../../examples/query_engine/sub_question_query_engine.ipynb
../../examples/query_transformations/SimpleIndexDemo-multistep.ipynb
../../examples/query_engine/SQLRouterQueryEngine.ipynb
../../examples/query_engine/SQLAutoVectorQueryEngine.ipynb
```