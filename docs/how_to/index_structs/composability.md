# Composability


LlamaIndex offers **composability** of your indices, meaning that you can build indices on top of other indices. This allows you to more effectively index your entire document tree in order to feed custom knowledge to GPT.

Composability allows you to to define lower-level indices for each document, and higher-order indices over a collection of documents. To see how this works, imagine defining 1) a tree index for the text within each document, and 2) a list index over each tree index (one document) within your collection.

### Defining Subindices
To see how this works, imagine you have 3 documents: `doc1`, `doc2`, and `doc3`.

```python
from llama_index import SimpleDirectoryReader

doc1 = SimpleDirectoryReader('data1').load_data()
doc2 = SimpleDirectoryReader('data2').load_data()
doc3 = SimpleDirectoryReader('data3').load_data()
```

![](/_static/composability/diagram_b0.png)

Now let's define a tree index for each document. In order to persist the graph later, each index should share the same storage context.

In Python, we have:

```python
from llama_index import GPTTreeIndex

storage_context = storage_context.from_defaults()

index1 = GPTTreeIndex.from_documents(doc1, storage_context=storage_context)
index2 = GPTTreeIndex.from_documents(doc2, storage_context=storage_context)
index3 = GPTTreeIndex.from_documents(doc3, storage_context=storage_context)
```

![](/_static/composability/diagram_b1.png)

### Defining Summary Text

You then need to explicitly define *summary text* for each subindex. This allows  
the subindices to be used as Documents for higher-level indices.

```python
index1_summary = "<summary1>"
index2_summary = "<summary2>"
index3_summary = "<summary3>"
```

You may choose to manually specify the summary text, or use LlamaIndex itself to generate
a summary, for instance with the following:

```python
summary = index1.query(
    "What is a summary of this document?", retriever_mode="all_leaf"
)
index1_summary = str(summary)
```

**If specified**, this summary text for each subindex can be used to refine the answer during query-time. 

### Creating a Graph with a Top-Level Index

We can then create a graph with a list index on top of these 3 tree indices:
We can query, save, and load the graph to/from disk as any other index.

```python
from llama_index.indices.composability import ComposableGraph

graph = ComposableGraph.from_indices(
    GPTListIndex,
    [index1, index2, index3],
    index_summaries=[index1_summary, index2_summary, index3_summary],
    storage_context=storage_context,
)

```

![](/_static/composability/diagram.png)


### Querying the Graph

During a query, we would start with the top-level list index. Each node in the list corresponds to an underlying tree index. 
The query will be executed recursively, starting from the root index, then the sub-indices.
The default query engine for each index is called under the hood (i.e. `index.as_query_engine()`), unless otherwise configured by passing `custom_query_engines` to the `ComposableGraphQueryEngine`.
Below we show an example that configure the tree index retrievers to use `child_branch_factor=2` (instead of the default `child_branch_factor=1`).


More detail on how to configure `ComposableGraphQueryEngine` can be found [here](/reference/query/query_engines/graph_query_engine.rst).


```python
# set custom retrievers. An example is provided below
custom_query_engines = {
    index.index_id: index.as_query_engine(
        child_branch_factor=2
    ) 
    for index in [index1, index2, index3]
}
query_engine = graph.as_query_engine(
    custom_query_engines=custom_query_engines
)
response = query_engine.query("Where did the author grow up?")
```

> Note that specifying custom retriever for index by id
> might require you to inspect e.g., `index1.index_id`.
> Alternatively, you can explicitly set it as follows:
```python
index1.set_index_id("<index_id_1>")
index2.set_index_id("<index_id_2>")
index3.set_index_id("<index_id_3>")
```

![](/_static/composability/diagram_q1.png)

So within a node, instead of fetching the text, we would recursively query the stored tree index to retrieve our answer.

![](/_static/composability/diagram_q2.png)

NOTE: You can stack indices as many times as you want, depending on the hierarchies of your knowledge base! 


### [Optional] Persisting the Graph

The graph can also be persisted to storage, and then loaded again when needed. Note that you'll need to set the 
ID of the root index, or keep track of the default.

```python
# set the ID
graph.root_index.set_index_id("my_id")

# persist to storage
graph.root_index.storage_context.persist(persist_dir="./storage")

# load 
from llama_index import StorageContext, load_graph_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
graph = load_graph_from_storage(storage_context, root_id="my_id")
```


We can take a look at a code example below as well. We first build two tree indices, one over the Wikipedia NYC page, and the other over Paul Graham's essay. We then define a keyword extractor index over the two tree indices.

[Here is an example notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/composable_indices/ComposableIndices.ipynb).


```{toctree}
---
caption: Examples
maxdepth: 1
---
../../examples/composable_indices/ComposableIndices-Prior.ipynb
../../examples/composable_indices/ComposableIndices-Weaviate.ipynb
../../examples/composable_indices/ComposableIndices.ipynb
```