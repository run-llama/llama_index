# Deprecated Terms

As LlamaIndex continues to evolve, many class names and APIs have been adjusted, improved, and deprecated.

The following is a list of previously popular terms that have been deprecated, with links to their replacements.

## GPTSimpleVectorIndex

This has been renamed to `VectorStoreIndex`, as well as unifying all vector indexes to a single unified interface. You can integrate with various vector databases by modifying the underlying `vector_store`.

Please see the following links for more details on usage.

- [Index Usage Pattern](/docs/module_guides/evaluating/usage_pattern.md)
- [Vector Store Guide](/docs/module_guides/indexing/vector_store_guide.ipynb)
- [Vector Store Integrations](/docs/community/integrations/vector_stores.md)

## GPTVectorStoreIndex

This has been renamed to `VectorStoreIndex`, but it is only a cosmetic change. Please see the following links for more details on usage.

- [Index Usage Pattern](/docs/module_guides/evaluating/usage_pattern.md)
- [Vector Store Guide](/docs/module_guides/indexing/vector_store_guide.ipynb)
- [Vector Store Integrations](/docs/community/integrations/vector_stores.md)

## LLMPredictor

The `LLMPredictor` object is no longer intended to be used by users. Instead, you can setup an LLM directly and pass it into the `ServiceContext`. The `LLM` class itself has similar attributes and methods as the `LLMPredictor`.

- [LLMs in LlamaIndex](/docs/module_guides/models/llms.md)
- [Setting LLMs in the ServiceContext](/docs/module_guides/supporting_modules/service_context.md)

## PromptHelper and max_input_size/

The `max_input_size` parameter for the prompt helper has since been replaced with `context_window`.

The `PromptHelper` in general has been deprecated in favour of specifying parameters directly in the `service_context` and `node_parser`.

See the following links for more details.

- [Configuring settings in the Service Context](/docs/module_guides/supporting_modules/service_context.md)
- [Parsing Documents into Nodes](/docs/module_guides/loading/node_parsers/root.md)
