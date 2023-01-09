# GPT Index Use Cases
GPT Index data structures and parameters offer distinct use cases and advantages.
This guide should paint a picture of how you can use GPT Index to solve your own data needs. 
We go through each use case, and describe the index tools you can use for each use case.

## By Use Cases

### Use Case: Just Starting Out
To start with, you will mostly likely want to use a [Vector Store Index](vector-store-index). 
Vector Store Indices
are a simple and effective tool that allows you to answer a query over a large corpus of data.
When you define a Vector Store Index over a collection of documents, it embeds each text chunk and stores the 
embedding in an underlying vector store. To answer a query, the vector store index embedds the query, 
fetches the top-k text chunks by embedding similarity, and runs the LLM over these chunks in order to synthesize the answer.
[The starter example](/getting_started/starter_example.md) shows how to get started using a Vector Store Index
(`GPTSimpleVectorIndex`). See [Embedding Support How-To](/how_to/embeddings.md) for a more detailed treatment of all vector
store indices (e.g. using Faiss, Weaviate).

Our Vector Store Indices are good to start with because they generalize to a broad variety of use cases. 
For a more detailed/advanced treatment of different use cases and how they map to indices, please see below.


### Use Case: Connecting GPT Index to an External Data Source of Documents

To connect GPT Index to a large external data source of documents, you will want to [use one of our data connectors](/how_to/data_connectors.md), or construct `Document` objects manually (see the [primer guide](/guides/primer.md) for how).

Then you will likely want to use a [Vector Store Index](vector-store-index).


### Use Case: Summarization over Documents

You want to perform a *summarization* query over your collection of documents. A summarization query requires GPT to iterate through many if not most documents in order to synthesize an answer.
For instance, a summarization query could look like one of the following: 
- "What is a summary of this collection of text?"
- "Give me a summary of person X's experience with the company."

You can use most indices e.g. a [Vector Store Index](vector-store-index), a list index (`GPTListIndex`)
to construct a summary with `response_mode="tree_summarize"`. See [here](/guides/usage_pattern.md) for more details on response modes.

```python
index = GPTListIndex(documents)

response = index.query("<summarization_query>", response_mode="tree_summarize")
```

You can also construct a summary using the tree index (`GPTTreeIndex`), but using the `mode` parameter instead:

```python
index = GPTTreeIndex(documents)

response = index.query("<summarization_query>", mode="summarize")
```

This is because the "default" mode for a tree-based query is traversing from the top of the graph down to leaf nodes. For summarization purposes we will want
to use `mode="summarize"` instead.


### Use Case: Combining information across multiple indices

You have information in two or more different data sources (e.g. Notion and Slack). 
You want to explicitly synthesize an answer combining information in these data sources.

While a single vector store may implicitly do so (the top-k nearest neighbor text chunks could be from Notion or Slack), a better data structure for explicitly doing this would be the List Index `GPTListIndex`.

Assuming you've already defined "subindices" over each data source, you can define a higher-level list index on top of these subindices through [composability](/how_to/composability.md).

```python
from gpt_index import GPTSimpleVectorIndex, GPTListIndex

index1 = GPTSimpleVectorIndex(notion_docs)
index2 = GPTSimpleVectorIndex(slack_docs)

# Set summary text
# you can set the summary manually, or you can
# generate the summary itself using GPT Index
index1.set_text("summary1")
index2.set_text("summary2")

index3 = GPTListIndex([index1, index2])

response = index3.query("<query_str>", mode="recursive", query_configs=...)

```


### Use Case: Routing a Query to the Right Index

You have a few disparate data sources, represented as Document objects
or subindices. You want to "route" a query to an underlying Document or a subindex.
Here you have three options: `GPTTreeIndex`, `GPTKeywordTableIndex`, or a
[Vector Store Index](vector-store-index).

A `GPTTreeIndex` uses the LLM to select the child node(s) to send the query down to.
A `GPTKeywordTableIndex` uses keyword matching, and a `GPTVectorStoreIndex` uses
embedding cosine similarity.

```python
from gpt_index import GPTTreeIndex, GPTSimpleVectorIndex

...

# subindices
index1 = GPTSimpleVectorIndex(notion_docs)
index2 = GPTSimpleVectorIndex(slack_docs)

# Set summary text
# you can set the summary manually, or you can
# generate the summary itself using GPT Index
index1.set_text("summary1")
index2.set_text("summary2")

# tree index for routing
tree_index = GPTTreeIndex([index1, index2])

response = tree_index.query(
    "In Notion, give me a summary of the product roadmap.",
    mode="recursive",
    query_configs=...
)

```


### Use Case: Using Keyword Filters

You want to explicitly filter nodes by keywords.
You can set `required_keywords` and `exclude_keywords` on most of our indices (the only exclusion is the GPTTreeIndex). This will preemptively filter out nodes that do not contain `required_keywords` or contain `exclude_keywords`, reducing the search space
and hence time/number of LLM calls/cost.

See the [Usage Pattern Guide](/guides/usage_pattern.md) around querying with required_keywords and exclude_keywords.


### Use Case: Including Hierarchical Context in your Answer

You have a knowledge base that is organized in a hierarchy. For instance, you may have a book that is organized at the top-level by chapter, and then within each chapter there is a large body of text. Or you have an product roadmap document that is first organized by top-level goals, and then organized by project. You want your answer to include both high-level context as well as details within the lower-level text.

You can do this by defining a subindex for each subsection, defining a *summary text* for that subindex, and [a higher order index](/how_to/composability.md) to combine the subindices. You can stack this as many times as you wish. By defining summary text for each subsection, the higher order index will *refine* the answer synthesized through a subindex with the summary.

```python
from gpt_index import GPTTreeIndex, GPTSimpleVectorIndex


index1 = GPTSimpleVectorIndex(chapter1)
index2 = GPTSimpleVectorIndex(chapter2)

# Set summary text
# you can set the summary manually, or you can
# generate the summary itself using GPT Index
index1.set_text("summary1")
index2.set_text("summary2")

# build tree index
index3 = GPTTreeIndex([index1, index2])

response = index3.query("<query_str>", mode="recursive", query_configs=...)

```

