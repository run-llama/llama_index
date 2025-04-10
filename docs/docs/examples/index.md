# Examples

LlamaIndex provides a rich collection of examples demonstrating diverse use cases, integrations, and features. This page highlights key examples to help you get started.

In the navigation to the left, you will also find many example notebooks, displaying the usage of various llama-index components and use-cases.

## Agents

Build powerful AI assistants with LlamaIndex's agent capabilities:

- [Function Calling Agent](./agent/agent_workflow_basic.ipynb) - Learn the basics of Function Calling Agents and `AgentWorkflow`
- [React Agent](./agent/react_agent.ipynb) - Use the ReAct (Reasoning and Acting) pattern with agents
- [Code Act Agent](./agent/code_act_agent.ipynb) - Agents that can write and execute code
- [Multi-Agent Workflow](./agent/agent_workflow_multi.ipynb) - Build a multi-agent workflow with `AgentWorkflow`

You might also be interested in the [general introduction to agents](../understanding/agent/index.md).

## Agentic Workflows

Use LlamaIndex Workflows to build agentic systems:

- [Function Calling Agent from Scratch](./workflow/function_calling_agent.ipynb) - Build a Function Calling agent from scratch
- [React Agent from Scratch](./workflow/react_agent_from_scratch.ipynb) - Build a ReAct agent from scratch
- [CodeAct Agent from Scratch](./agent/from_scratch_code_act_agent.ipynb) - Build a CodeAct agent from scratch
- [Basic RAG](./workflow/rag.ipynb) - Simple RAG workflow implementation
- [Advanced Text-to-SQL](./workflow/advanced_text_to_sql.ipynb) - Use LlamaIndex to generate SQL queries and execute them

You might also be interested in the [general introduction to agentic workflows](../understanding/workflows/index.md).

## LLM Integrations

Connect with popular LLM providers:

- [OpenAI](./llm/openai.ipynb) - Use OpenAI models (GPT-3.5, GPT-4, etc.)
- [Anthropic](./llm/anthropic.ipynb) - Integrate with Claude models
- [Bedrock](./llm/bedrock_converse.ipynb) - Work with Meta's Llama 3 models
- [Gemini/Vertex](./llm/gemini.ipynb) - Use Google's Gemini/Vertex models
- [Mistral](./llm/mistralai.ipynb) - Integrate with Mistral AI models
- [Ollama](./llm/ollama.ipynb) - Use Ollama models locally

You might also be interested in the [general introduction to LLM in LlamaIndex](../understanding/using_llms/using_llms.md).

## Embedding Models

Various embedding model integrations:

- [OpenAI Embeddings](./embeddings/OpenAI.ipynb) - OpenAI's text embedding models
- [Cohere Embeddings](./embeddings/cohereai.ipynb) - Cohere's embedding models
- [HuggingFace Embeddings](./embeddings/huggingface.ipynb) - Use open-source embeddings from HuggingFace locally
- [Jina Embeddings](./embeddings/jina_embeddings.ipynb) - Jina AI's embedding models
- [Ollama Embeddings](./embeddings/ollama_embedding.ipynb) - Ollama's embedding models

## Vector Stores

Store and retrieve vector embeddings:

- [Pinecone](./vector_stores/PineconeIndexDemo.ipynb) - Pinecone vector database integration
- [Chroma](./vector_stores/ChromaIndexDemo.ipynb) - Chroma vector store
- [Weaviate](./vector_stores/WeaviateIndexDemo.ipynb) - Weaviate vector database
- [Qdrant](./vector_stores/QdrantIndexDemo.ipynb) - Qdrant vector database
- [MongoDB Atlas](./vector_stores/MongoDBAtlasVectorSearch.ipynb) - MongoDB Atlas Vector Search
- [Redis](./vector_stores/RedisIndexDemo.ipynb) - Redis vector database
- [Milvus](./vector_stores/MilvusIndexDemo.ipynb) - Milvus vector database
- [Azure AI Search](./vector_stores/AzureAISearchIndexDemo.ipynb) - Azure AI Search vector database

You might also be interested in the [general introduction to vector stores and retrieval](../understanding/rag/index.md).
