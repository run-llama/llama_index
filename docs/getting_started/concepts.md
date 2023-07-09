# High-Level Concepts

Okay now that you've experienced the magic of asking question over your data (if not, you should start with [the quick start tutorial](/docs/getting_started/starter_example.md)). you might be wondering: how does this all work?

This guide give you:
* high-level understanding of what's going on under the hood,
* and explain the key concepts in LlamaIndex necessary,
* help you further configure the pipeline to get better performance.

## Retrieval Augmented Generation (RAG)
LlamaIndex is a data framework that let you build retrieval-augmented generation (RAG) based LLM applications super easily.  
**RAG** is a paradigm for augmenting LLM with your data.
It involves 1) preparing a knowledge base, and 2) first retrieving data from some knowledge base, and then feed it to the LLM for generating an answer.
![](/_static/getting_started/rag.png)

To setup this pipeline: there are two main challenges: 
1) preparing the knowledge base to efficiently retrieve relevant context, and 
2) defining the query engine that combines retrieved context and synthesizing a response/message with a LLM.

LlamaIndex helps make both steps super easy.
Let's explore each stage in detail and the associated LlamaIndex modules to make it super easy.

### Stage 1: Prepare the knowledge base 
LlamaIndex help you prepare the knowledge with a suite of data connectors and indexes.
![](/_static/getting_started/indexing.png) 

[**Data Loaders**](/core_modules/data_modules/connector/root.md):
A data connector (i.e. `Reader`) ingest data from different data sources and data formats into a simple `Document` representation (text and simple metadata).


[**Data Indexes**](/core_modules/data_modules/index/root.md): 
Once you've ingested your data, LlamaIndex help you index data into a format that's easy to retrieve.

### Stage 2: Query against knowledge base
Now that we have an index over the documents, we can efficiently retrieve relevant information
to a user question or in a conversation.
![](/_static/getting_started/querying.png)

[**Retriever**](/core_modules//query_modules/retriever/root.md): 
retrieve relevant documents from indices

[**Query Engine**](/core_modules/query_modules/query_engine/root.md):


[**Chat Engine**](/core_modules/query_modules/chat_engines/root.md): 
have a conversation with your data

[**Agent**](/core_modules/query_modules/agent/root.md): 
add additional tools and trigger workflow