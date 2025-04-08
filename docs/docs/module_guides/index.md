# Component Guides

Welcome to the LlamaIndex component guides! This section provides detailed documentation for all the core modules and components of the LlamaIndex framework.

## Core Components

### Models
- [Introduction to Models](./models/index.md) - Overview of model components
- [LLMs](./models/llms.md) - Language models for text generation and reasoning
- [Embeddings](./models/embeddings.md) - Convert text to vector representations
- [Multi Modal](./models/multi_modal.md) - Work with images, audio, and other non-text data

### Prompts
- [Introduction to Prompts](./models/prompts/index.md) - Overview of prompt engineering
- [Usage Patterns](./models/prompts/usage_pattern.md) - Learn how to effectively use prompts

### Loading
- [Introduction to Loading](./loading/index.md) - Overview of data loading capabilities
- [Documents and Nodes](./loading/documents_and_nodes/index.md) - Core data structures
- [SimpleDirectoryReader](./loading/simpledirectoryreader.md) - Easy document loading
- [Data Connectors](./loading/connector/index.md) - Connect to external data sources
- [Node Parsers / Text Splitters](./loading/node_parsers/index.md) - Split documents into chunks
- [Ingestion Pipeline](./loading/ingestion_pipeline/index.md) - End-to-end document processing

### Indexing
- [Introduction to Indexing](./indexing/index.md) - Overview of indexing approaches
- [Index Guide](./indexing/index_guide.md) - Comprehensive guide to indices
- [Vector Store Index](./indexing/vector_store_index.md) - Semantic search with vectors
- [Property Graph Index](./indexing/lpg_index_guide.md) - Graph-based indexing

### Storing
- [Introduction to Storing](./storing/index.md) - Overview of storage components
- [Vector Stores](./storing/vector_stores.md) - Store embeddings for retrieval
- [Document Stores](./storing/docstores.md) - Persist document collections
- [Index Stores](./storing/index_stores.md) - Store index metadata

### Querying
- [Introduction to Querying](./querying/index.md) - Overview of query components
- [Query Engines](./deploying/query_engine/index.md) - Process and answer queries
- [Chat Engines](./deploying/chat_engines/index.md) - Build conversational interfaces
- [Retrieval](./querying/retriever/index.md) - Retrieve relevant context
- [Response Synthesis](./querying/response_synthesizers/index.md) - Generate coherent answers

## Advanced Components

### Agents
- [Introduction to Agents](./deploying/agents/index.md) - Overview of agent capabilities
- [Memory](./deploying/agents/memory.md) - Add conversational memory to agents
- [Tools](./deploying/agents/tools.md) - Extend capabilities with external tools

### Workflows
- [Introduction to Workflows](./workflow/index.md) - Build complex, multi-step AI workflows

### Evaluation
- [Introduction to Evaluation](./evaluating/index.md) - Overview of evaluation frameworks
- [Usage Patterns](./evaluating/usage_pattern.md) - Test and improve your applications
- [LlamaDatasets](./evaluating/contributing_llamadatasets.md) - Standardized evaluation datasets

### Observability
- [Introduction to Observability](./observability/index.md) - Overview of monitoring capabilities
- [Instrumentation](./observability/instrumentation.md) - Monitor and debug your applications

### Settings
- [Settings Configuration](./supporting_modules/settings.md) - Configure global LlamaIndex settings

### Llama Deploy
- [Introduction to Llama Deploy](./llama_deploy) - Deploy LlamaIndex applications to production
