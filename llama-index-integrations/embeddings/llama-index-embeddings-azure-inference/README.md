# LlamaIndex Embeddings Integration: Azure AI model inference

The integration package `llama-index-embeddings-azure-inference` brings support for embedding models deployed in Azure AI, including Azure AI studio and Azure Machine Learning, to the `llama_index` ecosystem. Any model endpoint supporting the [Azure AI model inference API](https://aka.ms/azureai/modelinference) can be used with this integration.

For details and examples about how to use this integration, see [Getting starting with LlamaIndex and Azure AI](https://aka.ms/azureai/llamaindex).

## Changelog

- **0.2.4**:

  - Introduce `api_version` parameter in the `AzureAIEmbeddingsModel` class to allow overriding of the default value.
  - Introduce support for [Azure AI model inference service](https://aka.ms/aiservices/infernece) which requires `azure-ai-inference>=1.0.0b5`.
