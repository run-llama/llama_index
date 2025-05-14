# Deprecated Terms

As LlamaIndex continues to evolve, many class names and APIs have been adjusted, improved, and deprecated.

The following is a list of previously popular terms that have been deprecated, with links to their replacements.

## GPTSimpleVectorIndex

This has been renamed to `VectorStoreIndex`, as well as unifying all vector indexes to a single unified interface. You can integrate with various vector databases by modifying the underlying `vector_store`.

Please see the following links for more details on usage.

- [Index Usage Pattern](../module_guides/evaluating/usage_pattern.md)
- [Vector Store Guide](../module_guides/indexing/vector_store_guide.ipynb)
- [Vector Store Integrations](../community/integrations/vector_stores.md)

## GPTVectorStoreIndex

This has been renamed to `VectorStoreIndex`, but it is only a cosmetic change. Please see the following links for more details on usage.

- [Index Usage Pattern](../module_guides/evaluating/usage_pattern.md)
- [Vector Store Guide](../module_guides/indexing/vector_store_guide.ipynb)
- [Vector Store Integrations](../community/integrations/vector_stores.md)

## LLMPredictor

The `LLMPredictor` object is no longer intended to be used by users. Instead, you can setup an LLM directly and pass it into the `Settings` or the interface using the LLM. The `LLM` class itself has similar attributes and methods as the `LLMPredictor`.

- [LLMs in LlamaIndex](../module_guides/models/llms.md)
- [Setting LLMs in the Settings](../module_guides/supporting_modules/settings.md)

## PromptHelper and max_input_size/

The `max_input_size` parameter for the prompt helper has since been replaced with `context_window`.

The `PromptHelper` in general has been deprecated in favour of specifying parameters directly in the `service_context` and `node_parser`.

See the following links for more details.

- [Configuring settings in the Settings](../module_guides/supporting_modules/settings.md)
- [Parsing Documents into Nodes](../module_guides/loading/node_parsers/index.md)

## ServiceContext

The `ServiceContext` object has been deprecated in favour of the `Settings` object.

- [Configuring settings in the Settings](../module_guides/supporting_modules/settings.md)

## llama-index-legacy

The `llama-index-legacy` package has been deprecated and removed from the repository. Please see the latest getting started guide for the latest information and usage.

- [Getting Started](../getting_started/installation.md)

## AgentRunner/AgentWorker (and related classes)

The `AgentRunner` and `AgentWorker` classes have been deprecated in favour of [AgentWorkflow](../module_guides/deploying/agents/index.md) and [Workflows](../module_guides/workflow/index.md).

This includes the following deprecated classes:

- `AgentRunner`
- `FunctionCallingAgent`
- `FunctionCallingAgentWorker`
- `llama_index.core.agent.ReActAgent` (use [llama_index.core.agent.workflow.ReActAgent](../module_guides/workflow/index.md))
- `ReActAgentWorker`
- `LATSAgentWorker`
- `CoAAgentWorker`
- `FnAgentWorker`
- `QueryPipelineAgentWorker`
- `MultiModalReActAgentWorker`
- `IntrospectiveAgentWorker`
- `SelfReflectiveAgentWorker`
- `ToolInteractiveReflectionAgentWorker`
- `LLMCompilerAgentWorker`
- `QueryUnderstandAgentWorker`
