# Queries over your Data

At a high-level, LlamaIndex gives you the ability to query your data for any downstream LLM use case,
whether it's question-answering, summarization, or a component in a chatbot.

This section describes the different ways you can query your data with LlamaIndex, roughly in order
of simplest (top-k semantic search), to more advanced capabilities.

### Semantic Search 

The most basic example usage of LlamaIndex is through semantic search. We provide
a simple in-memory vector store for you to get started, but you can also choose
to use any one of our [vector store integrations](/how_to/integrations/vector_stores.md):

```python
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader('data').load_data()
index = GPTSimpleVectorIndex.from_documents(documents)
response = index.query("What did the author do growing up?")
print(response)

```

Relevant Resources:
- [Quickstart](/getting_started/starter_example.md)
- [Example notebook](https://github.com/jerryjliu/llama_index/tree/main/examples/vector_indices)


### Summarization

A summarization query requires the LLM to iterate through many if not most documents in order to synthesize an answer.
For instance, a summarization query could look like one of the following: 
- "What is a summary of this collection of text?"
- "Give me a summary of person X's experience with the company."

In general, a list index would be suited for this use case. A list index by default goes through all the data.

Empirically, setting `response_mode="tree_summarize"` also leads to better summarization results.

```python
index = GPTListIndex.from_documents(documents)

response = index.query("<summarization_query>", response_mode="tree_summarize")
```

### Queries over Structured Data

LlamaIndex supports queries over structured data, whether that's a Pandas DataFrame or a SQL Database.

Here are some relevant resources:
- [Guide on Text-to-SQL](/guides/tutorials/sql_guide.md)
- [SQL Demo Notebook 1](https://github.com/jerryjliu/llama_index/blob/main/examples/struct_indices/SQLIndexDemo.ipynb)
- [SQL Demo Notebook 2 (Context)](https://github.com/jerryjliu/llama_index/blob/main/examples/struct_indices/SQLIndexDemo-Context.ipynb)
- [SQL Demo Notebook 3 (Big tables)](https://github.com/jerryjliu/llama_index/blob/main/examples/struct_indices/SQLIndexDemo-ManyTables.ipynb)
- [Pandas Demo Notebook](https://github.com/jerryjliu/llama_index/blob/main/examples/struct_indices/PandasIndexDemo.ipynb).


### Synthesis over Heterogenous Data

LlamaIndex supports synthesizing across heterogenous data sources. This can be done by composing a graph over your existing data.
Specifically, compose a list index over your subindices. A list index inherently combines information for each node; therefore
it can synthesize information across your heteregenous data sources.

```python
from llama_index import GPTSimpleVectorIndex, GPTListIndex
from llama_index.indices.composability import ComposableGraph

index1 = GPTSimpleVectorIndex.from_documents(notion_docs)
index2 = GPTSimpleVectorIndex.from_documents(slack_docs)

graph = ComposableGraph.from_indices(GPTListIndex, [index1, index2], index_summaries=["summary1", "summary2"])
response = graph.query("<query_str>", mode="recursive", query_configs=...)

```

Here are some relevant resources:
- [Composability](/how_to/index_structs/composability.md)
- [City Analysis Demo](https://github.com/jerryjliu/llama_index/blob/main/examples/composable_indices/city_analysis/PineconeDemo-CityAnalysis.ipynb).



### Routing over Heterogenous Data

LlamaIndex also supports routing over heteregenous data sources - for instance, if you want to "route" a query to an 
underlying Document or a subindex.
Here you have three options: `GPTTreeIndex`, `GPTKeywordTableIndex`, or a
[Vector Store Index](vector-store-index).

A `GPTTreeIndex` uses the LLM to select the child node(s) to send the query down to.
A `GPTKeywordTableIndex` uses keyword matching, and a `GPTVectorStoreIndex` uses
embedding cosine similarity.

```python
from llama_index import GPTTreeIndex, GPTSimpleVectorIndex
from llama_index.indices.composability import ComposableGraph

...

# subindices
index1 = GPTSimpleVectorIndex.from_documents(notion_docs)
index2 = GPTSimpleVectorIndex.from_documents(slack_docs)

# tree index for routing
tree_index = ComposableGraph.from_indices(
    GPTTreeIndex, 
    [index1, index2],
    index_summaries=["summary1", "summary2"]
)

response = tree_index.query(
    "In Notion, give me a summary of the product roadmap.",
    mode="recursive",
    query_configs=...
)

```

Here are some relevant resources:
- [Composability](/how_to/index_structs/composability.md)
- [Composable Keyword Table Graph](https://github.com/jerryjliu/llama_index/blob/main/examples/composable_indices/ComposableIndices.ipynb).



### Compare/Contrast Queries

LlamaIndex can support compare/contrast queries as well. It can do this in the following fashion:
- Composing a graph over your data
- Adding in query transformations.


You can perform compare/contrast queries by just composing a graph over your data.

Here are some relevant resources:
- [Composability](/how_to/index_structs/composability.md)
- [SEC 10-k Analysis Example notebook](https://colab.research.google.com/drive/1uL1TdMbR4kqa0Ksrd_Of_jWSxWt1ia7o?usp=sharing).


You can also perform compare/contrast queries with a **query transformation** module.

```python
from gpt_index.indices.query.query_transform.base import DecomposeQueryTransform
decompose_transform = DecomposeQueryTransform(
    llm_predictor_chatgpt, verbose=True
)
```

This module will help break down a complex query into a simpler one over your existing index structure.

Here are some relevant resources:
- [Query Transformations](/how_to/query/query_transformations.md)
- [City Analysis Example Notebook](https://github.com/jerryjliu/llama_index/blob/main/examples/composable_indices/city_analysis/City_Analysis-Decompose.ipynb)


### Multi-Step Queries

LlamaIndex can also support multi-step queries. Given a complex query, break it down into subquestions.

For instance, given a question "Who was in the first batch of the accelerator program the author started?",
the module will first decompose the query into a simpler initial question "What was the accelerator program the author started?",
query the index, and then ask followup questions.

Here are some relevant resources:
- [Query Transformations](/how_to/query/query_transformations.md)
- [Multi-Step Query Decomposition Notebook](https://github.com/jerryjliu/llama_index/blob/main/examples/vector_indices/SimpleIndexDemo-multistep.ipynb)




