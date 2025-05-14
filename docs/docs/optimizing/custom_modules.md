# Writing Custom Modules

A core design principle of LlamaIndex is that **almost every core module can be subclassed and customized**.

This allows you to use LlamaIndex for any advanced LLM use case, beyond the capabilities offered by our prepackaged modules. You're free to write as much custom code for any given module, but still take advantage of our lower-level abstractions and also plug this module along with other components.

We offer convenient/guided ways to subclass our modules, letting you write your custom logic without having to worry about having to define all boilerplate (for instance, [callbacks](../module_guides/observability/callbacks/index.md)).

This guide centralizes all the resources around writing custom modules in LlamaIndex. Check them out below 👇

## Custom LLMs

- [Custom LLMs](../module_guides/models/llms/usage_custom.md#customizing-llms-within-llamaindex-abstractions)

## Custom Embeddings

- [Custom Embedding Model](../module_guides/models/embeddings.md#custom-embedding-model)

## Custom Output Parsers

- [Custom Output Parsers](../examples/output_parsing/llm_program.ipynb)

## Custom Transformations

- [Custom Transformations](../module_guides/loading/ingestion_pipeline/transformations.md#custom-transformations)
- [Custom Property Graph Extractors](../module_guides/indexing/lpg_index_guide.md#sub-classing-extractors)

## Custom Retrievers

- [Custom Retrievers](../examples/query_engine/CustomRetrievers.ipynb)
- [Custom Property Graph Retrievers](../module_guides/indexing/lpg_index_guide.md#sub-classing-retrievers)

## Custom Postprocessors/Rerankers

- [Custom Node Postprocessor](./custom_modules.md#custom-postprocessorsrerankers)

## Custom Query Engines

- [Custom Query Engine](../examples/query_engine/custom_query_engine.ipynb)

## Custom Agents

- [Custom Function Calling Agent](../examples/workflow/function_calling_agent.ipynb)
- [Custom ReAct Agent](../examples/workflow/react_agent.ipynb)

## Custom Query Components (for use in Query Pipeline)

- [Custom Query Component](../module_guides/querying/pipeline/usage_pattern.md#defining-a-custom-query-component)

## Other Ways of Customization

Some modules can be customized heavily within your workflows but not through subclassing (and instead through parameters or functions we expose). We list these in guides below:

- [Customizing Documents](../module_guides/loading/documents_and_nodes/usage_documents.md)
- [Customizing Nodes](../module_guides/loading/documents_and_nodes/usage_nodes.md)
- [Customizing Prompts within Higher-Level Modules](../examples/prompts/prompt_mixin.ipynb)
