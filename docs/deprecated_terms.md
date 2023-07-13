# Deprecated Terms

As LlamaIndex continues to evolve, many class names and APIs have been adjusted, improved, and deprecated.

The following is a list of previously popular terms that have been deprecated, with links to their replacements.

## GPTSimpleVectorIndex

This has been renamed to `VectorStoreIndex`, as well as unifying all vector indexes to a single unified interface. You can integrate with various vector databases by modifying the underlying `vector_store`. 

Please see the following links for more details on usage.

- [Index Usage Pattern](/core_modules/data_modules/index/usage_pattern.md)
- [Vector Store Guide](/core_modules/data_modules/index/vector_store_guide.ipynb)
- [Vector Store Integrations](/community/integrations/vector_stores.md)

## GPTVectorStoreIndex

This has been renamed to `VectorStoreIndex`, but it is only a cosmetic change. Please see the following links for more details on usage.

- [Index Usage Pattern](/core_modules/data_modules/index/usage_pattern.md)
- [Vector Store Guide](/core_modules/data_modules/index/vector_store_guide.ipynb)
- [Vector Store Integrations](/community/integrations/vector_stores.md)

## LLMPredictor

The `LLMPredictor` object is no longer intended to be used by users. Instead, you can setup an LLM directly and pass it into the `ServiceContext`.

- [LLMs in LlamaIndex](/core_modules/model_modules/llms/root.md)
- [Setting LLMs in the ServiceContext](/core_modules/supporting_modules/service_context.md)

## PromptHelper and max_input_size/

The `max_input_size` parameter for the prompt helper has since been replaced with `context_window`.

The `PromptHelper` in general has been deprecated in favour of specifying parameters directly in the `service_context` and `node_parser`.

See the following links for more details.

- [Configuring settings in the Service Context](/core_modules/supporting_modules/service_context.md)
- [Parsing Documents into Nodes](/core_modules/data_modules/node_parsers/root.md)
