# A Primer to using GPT Index

At its core, GPT Index contains a toolkit of **index data structures** designed to easily connect LLM's with your external data.
GPT Index helps to provide the following advantages:
- Remove concerns over prompt size limitations.
- Abstract common usage patterns to reduce boilerplate code in your LLM app.
- Provide data connectors to your common data sources (Google Docs, Slack, etc.).
- Provide cost transparency + tools that reduce cost while increasing performance.


Each data structure offers distinct use cases and a variety of customizable parameters. These indices can then be 
*queried* in a general purpose manner, in order to achieve any task that you would typically achieve with an LLM:
- Question-Answering
- Summarization
- Text Generation (Stories, TODO's, emails, etc.)
- and more!

This primer is intended to help you get the most out of GPT Index. It gives a high-level overview of the following: 
1. How to quickly get started using GPT Index.
2. The general usage pattern of GPT Index.
3. [Advanced] Mapping Use Cases to GPT Index data Structures
4. FAQ

## Quickly Getting Started with GPT Index
To start with, you will want to use a [Vector Store Index](/how_to/vector_stores.md). Vector Store Indices
are a simple and effective tool that allows you to answer a query over a large corpus of data.
When you define a Vector Store Index over a collection of documents, it embeds each text chunk and stores the 
embedding in an underlying vector store. To answer a query, the vector store index embedds the query, 
fetches the top-k text chunks by embedding similarity, and runs the LLM over these chunks in order to synthesize the answer.

[The starter example](/getting_started/starter_example.md) shows how to get started using a Vector Store Index
(`GPTSimpleVectorIndex`). See [Embedding Support How-To](/how_to/embeddings.md) for a more detailed treatment of all vector
store indices (e.g. using Faiss, Weaviate).

Our Vector Store Indices are good to start with because they generalize to a broad variety of use cases. 
For a more detailed/advanced treatment of different use cases and how they map to indices, please see below.


## General Usage Pattern of GPT Index

The general usage pattern of GPT Index is as follows:
1. Load in documents (either manually, or through a data loader).
2. Build an index over the documents.
3. [Optional, Advanced] Building indices on top of other indices
3. [Optional] Save the index for future use.
4. Query the index.


### 1. Load in Documents

The first step is to load in data. This data is represented in the form of `Document` objects. 
We provide a variety of [data loaders](/how_to/data_connectors.md) which will load in Documents
through the `load_data` function, e.g.:

```python
from gpt_index import SimpleDirectoryReader

documents = SimpleDirectoryReader('data').load_data()

```

You can also choose to construct documents manually. GPT Index exposes the `Document` struct.

```python
from gpt_index import Document

text_list = [text1, text2, ...]
documents = [Document(t) for t in text_list]
```

### 2. Build an index over the Documents

We can now build an index over these Document objects. The simplest is to load in the Document objects during index initialization.

```python
from gpt_index import GPTSimpleVectorIndex

index = GPTSimpleVectorIndex(documents)

```

Depending on which index you use, GPT Index may make LLM calls in order to build the index.
Token usage can be tracked through the output generated after the `__init__` statement.

#### Inserting Documents

You can also take advantage of the `insert` capability and insert the 

#### Customizing LLM's

#### Customizing Prompts


#### Customizing embeddings


#### Cost Predictor



### 3. [Optional] Save the index for future use

### 4. Query the index.



## [Advanced] Mapping Use Cases to GPT Index data Structures







