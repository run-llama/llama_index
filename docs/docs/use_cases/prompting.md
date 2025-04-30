# Prompting

Prompting LLMs is a fundamental unit of any LLM application. You can build an entire application entirely around prompting, or orchestrate with other modules (e.g. retrieval) to build RAG, agents, and more.

LlamaIndex supports LLM abstractions and simple-to-advanced prompt abstractions to make complex prompt workflows possible.

## LLM Integrations

LlamaIndex supports 40+ LLM integrations, from proprietary model providers like OpenAI, Anthropic to open-source models/model providers like Mistral, Ollama, Replicate. It provides all the tools to standardize interface around common LLM usage patterns, including but not limited to async, streaming, function calling.

Here's the [full module guide for LLMs](../module_guides/models/llms.md).

## Prompts

LlamaIndex has robust prompt abstractions that capture all the common interaction patterns with LLMs.

Here's the [full module guide for prompts](../module_guides/models/prompts/index.md).

### Table Stakes
- [Text Completion Prompts](../examples/customization/prompts/completion_prompts.ipynb)
- [Chat Prompts](../examples/customization/prompts/chat_prompts.ipynb)

### Advanced
- [Variable Mappings, Functions, Partials](../examples/prompts/advanced_prompts.ipynb)
- [RichPromptTemplate Features](../../../examples/prompts/rich_prompt_template_features.ipynb)

## Prompt Chains and Pipelines

LlamaIndex has robust abstractions for creating sequential prompt chains, as well as general DAGs to orchestrate prompts with any other component. This allows you to build complex workflows, including RAG with multi-hop query understanding layers, as well as agents.

These pipelines are integrated with [observability partners](../module_guides/observability/index.md) out of the box.

The central guide for prompt chains and pipelines is through our [Query Pipelines](../module_guides/querying/pipeline/index.md).
