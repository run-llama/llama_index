---
title: Component Guides
---

Welcome to the LlamaIndex component guides! This section provides detailed documentation for all the core modules and components of the LlamaIndex framework.

## Core Components

### Models

- [Introduction to Models](/python/framework/module_guides/models) - Overview of model components
- [LLMs](/python/framework/module_guides/models/llms) - Language models for text generation and reasoning
- [Embeddings](/python/framework/module_guides/models/embeddings) - Convert text to vector representations
- [Multi Modal](/python/framework/module_guides/models/multi_modal) - Work with images, audio, and other non-text data

### Prompts

- [Introduction to Prompts](/python/framework/module_guides/models/prompts) - Overview of prompt engineering
- [Usage Patterns](/python/framework/module_guides/models/prompts/usage_pattern) - Learn how to effectively use prompts

### Loading

- [Introduction to Loading](/python/framework/module_guides/loading) - Overview of data loading capabilities
- [Documents and Nodes](/python/framework/module_guides/loading/documents_and_nodes) - Core data structures
- [SimpleDirectoryReader](/python/framework/module_guides/loading/simpledirectoryreader) - Easy document loading
- [Data Connectors](/python/framework/module_guides/loading/connector) - Connect to external data sources
- [Node Parsers / Text Splitters](/python/framework/module_guides/loading/node_parsers) - Split documents into chunks
- [Ingestion Pipeline](/python/framework/module_guides/loading/ingestion_pipeline) - End-to-end document processing

### Indexing

- [Introduction to Indexing](/python/framework/module_guides/indexing) - Overview of indexing approaches
- [Index Guide](/python/framework/module_guides/indexing/index_guide) - Comprehensive guide to indices
- [Vector Store Index](/python/framework/module_guides/indexing/vector_store_index) - Semantic search with vectors
- [Property Graph Index](/python/framework/module_guides/indexing/lpg_index_guide) - Graph-based indexing

### Storing

- [Introduction to Storing](/python/framework/module_guides/storing) - Overview of storage components
- [Vector Stores](/python/framework/module_guides/storing/vector_stores) - Store embeddings for retrieval
- [Document Stores](/python/framework/module_guides/storing/docstores) - Persist document collections
- [Index Stores](/python/framework/module_guides/storing/index_stores) - Store index metadata

### Querying

- [Introduction to Querying](/python/framework/module_guides/querying) - Overview of query components
- [Query Engines](/python/framework/module_guides/deploying/query_engine) - Process and answer queries
- [Chat Engines](/python/framework/module_guides/deploying/chat_engines) - Build conversational interfaces
- [Retrieval](/python/framework/module_guides/querying/retriever) - Retrieve relevant context
- [Response Synthesis](/python/framework/module_guides/querying/response_synthesizers) - Generate coherent answers

## Advanced Components

### Agents

- [Introduction to Agents](/python/framework/module_guides/deploying/agents) - Overview of agent capabilities
- [Memory](/python/framework/module_guides/deploying/agents/memory) - Add conversational memory to agents
- [Tools](/python/framework/module_guides/deploying/agents/tools) - Extend capabilities with external tools

### Workflows

- [Introduction to Workflows](/python/framework/module_guides/workflow) - Build complex, multi-step AI workflows

### Evaluation

- [Introduction to Evaluation](/python/framework/module_guides/evaluating) - Overview of evaluation frameworks
- [Usage Patterns](/python/framework/module_guides/evaluating/usage_pattern) - Test and improve your applications
- [LlamaDatasets](/python/framework/module_guides/evaluating/contributing_llamadatasets) - Standardized evaluation datasets

### Observability

- [Introduction to Observability](/python/framework/module_guides/observability) - Overview of monitoring capabilities
- [Instrumentation](/python/framework/module_guides/observability/instrumentation) - Monitor and debug your applications

### Settings

- [Settings Configuration](/python/framework/module_guides/supporting_modules/settings) - Configure global LlamaIndex settings
