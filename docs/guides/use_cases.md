# GPT Index Use Cases

GPT Index data structures and parameters offer distinct use cases and advantages.
This guide should paint a picture of how you can use GPT Index to solve your own data needs. 
We go through each use case, and describe the index tools you can use for each use case.

## By Use Cases

### Use Case: Just Starting Out
To start with, you will mostly likely want to use a [Vector Store Index](/how_to/vector_stores.md). 
Vector Store Indices
are a simple and effective tool that allows you to answer a query over a large corpus of data.
When you define a Vector Store Index over a collection of documents, it embeds each text chunk and stores the 
embedding in an underlying vector store. To answer a query, the vector store index embedds the query, 
fetches the top-k text chunks by embedding similarity, and runs the LLM over these chunks in order to synthesize the answer.
to obtain
[The starter example](/getting_started/starter_example.md) shows how to get started using a Vector Store Index
(`GPTSimpleVectorIndex`). See [Embedding Support How-To](/how_to/embeddings.md) for a more detailed treatment of all vector
store indices (e.g. using Faiss, Weaviate).

Our Vector Store Indices are good to start with because they generalize to a broad variety of use cases. 
For a more detailed/advanced treatment of different use cases and how they map to indices, please see below.


### Use Case: Connecting GPT Index to an External Data Source of Documents


To connect GPT Index to a large external data source of documents, you will want to [use one of our data connectors](/how_to/data_connectors.md), or construct `Document` objects manually (see the [primer guide](/guides/primer.md) for how).

Then you will want to use a [Vector Store Index](/how_to/vector_stores.md).


### Use Case: Summarization over Documents

Say you want to perform a *summarization* query over your collection of documents. A summarization query requires GPT to iterate through many if not most documents in order to synthesize an answer.
For instance, a summarization query could look like one of the following: 
- "What is a summary of this collection of text?"
- "Give me a summary of person X's experience with the company."
You can use most indices e.g. a [Vector Store Index](/how_to/vector_stores.md), a list index (`GPTListIndex`), or a Tree Index (`GPTTreeIndex`)
to construct a summary with `response_mode="tree_summarize"`.

```python
index = GPTListIndex(documents)

index.query("<summarization_query>", response_mode="tree_summarize")
```



### Use Case: Combining information across multiple indices

Say you have information in two or more different data sources (e.g. Notion and Slack). 
You want to explicitly synthesize an answer combining information in these data sources.

While a single vector store may implicitly do so (the top-k nearest neighbor text chunks could be from Notion or Slack), a better data structure for explicitly doing this would be the List Index `GPTListIndex`.

Assuming you've already defined "subindices" over each data source, you can define a higher-level list index on top of these subindices through [composability](/how_to/composability.md).

```python
from gpt_index import GPTSimpleVectorIndex, GPTListIndex

index1 = GPTSimpleVectorIndex(notion_docs)
index2 = GPTSimpleVectorIndex(slack_docs)

index3 = GPTListIndex([index1, index2])

response = index3.query("<query_str>")

```


### Use Case: Routing a Query to the Right Index

Say you want to "route" a query to an underlying Document or a subindex.
Here you have three options: `GPTTreeIndex`, `GPTKeywordTableIndex`, or a
[Vector Store Index](/how_to/vector_stores.md).

A `GPTTreeIndex` uses the LLM to select the child node(s) to send the query down to.
A `GPTKeywordTableIndex` uses keyword matching, and a `GPTVectorStoreIndex` uses
embedding cosine similarity.

```python
from gpt_index import GPTTreeIndex, GPTSimpleVectorIndex

...

# subindices
index1 = GPTSimpleVectorIndex(notion_docs)
index2 = GPTSimpleVectorIndex(slack_docs)

# tree index for routing
tree_index = GPTTreeIndex([index1, index2])

response = tree_index.query("In Notion, give me a summary of the product roadmap.")

```

### Use Case: Using Keyword Filters

See TODO: Primer link in order to use keyword filters.
