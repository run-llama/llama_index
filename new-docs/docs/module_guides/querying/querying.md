# Querying

Querying is the most important part of your LLM application. To learn more about getting a final product that you can deploy, check out the [query engine](../deploying/query_engine/root.md), [chat engine](../deploying/chat_engines/root.md).

If you wish to combine advanced reasoning with tool use, check out our [agents](../deploying/agents/root.md) guide.

## Query Pipeline

You can create query pipelines/chains with ease with our declarative `QueryPipeline` interface. Check out our [query pipeline guide](pipeline/root.md) for more details.

```{toctree}
---
maxdepth: 1
hidden: True
---
pipeline/root.md
```

Otherwise check out how to use our query modules as standalone components 👇.

## Query Modules

```{toctree}
---
maxdepth: 1
---
../deploying/query_engine/root.md
../deploying/chat_engines/root.md
../deploying/agents/root.md
/module_guides/querying/retriever/root.md
/module_guides/querying/response_synthesizers/root.md
/module_guides/querying/router/root.md
/module_guides/querying/node_postprocessors/root.md
/module_guides/querying/structured_outputs/structured_outputs.md
```
