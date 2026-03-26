---
title: Writing Custom Modules
---

A core design principle of LlamaIndex is that **almost every core module can be subclassed and customized**.

This allows you to use LlamaIndex for any advanced LLM use case, beyond the capabilities offered by our prepackaged modules. You're free to write as much custom code for any given module, but still take advantage of our lower-level abstractions and also plug this module along with other components.

We offer convenient/guided ways to subclass our modules, letting you write your custom logic without having to worry about having to define all boilerplate (for instance, [callbacks](/python/framework/module_guides/observability/callbacks)).

This guide centralizes all the resources around writing custom modules in LlamaIndex. Check them out below 👇

## Custom LLMs

- [Custom LLMs](/python/framework/module_guides/models/llms/usage_custom#customizing-llms-within-llamaindex-abstractions)

## Custom Embeddings

- [Custom Embedding Model](/python/framework/module_guides/models/embeddings#custom-embedding-model)

## Custom Output Parsers

- [Custom Output Parsers](/python/examples/output_parsing/llm_program)

## Custom Transformations

- [Custom Transformations](/python/framework/module_guides/loading/ingestion_pipeline/transformations#custom-transformations)
- [Custom Property Graph Extractors](/python/framework/module_guides/indexing/lpg_index_guide#sub-classing-extractors)

## Custom Retrievers

- [Custom Retrievers](/python/examples/query_engine/customretrievers)
- [Custom Property Graph Retrievers](/python/framework/module_guides/indexing/lpg_index_guide#sub-classing-retrievers)

## Custom Postprocessors/Rerankers

- [Custom Node Postprocessor](/python/framework/optimizing/custom_modules#custom-postprocessorsrerankers)

## Custom Query Engines

- [Custom Query Engine](/python/examples/query_engine/custom_query_engine)

## Custom Agents

- [Custom Function Calling Agent](/python/examples/workflow/function_calling_agent)
- [Custom ReAct Agent](/python/examples/workflow/react_agent)

## Other Ways of Customization

Some modules can be customized heavily within your workflows but not through subclassing (and instead through parameters or functions we expose). We list these in guides below:

- [Customizing Documents](/python/framework/module_guides/loading/documents_and_nodes/usage_documents)
- [Customizing Nodes](/python/framework/module_guides/loading/documents_and_nodes/usage_nodes)
- [Customizing Prompts within Higher-Level Modules](/python/examples/prompts/prompt_mixin)
