# Composability


LlamaIndex offers **composability** of your indices, meaning that you can build indices on top of other indices. This allows you to more effectively index your entire document tree in order to feed custom knowledge to GPT.

Composability allows you to to define lower-level indices for each document, and higher-order indices over a collection of documents. To see how this works, imagine defining 1) a tree index for the text within each document, and 2) a list index over each tree index (one document) within your collection.

### Defining Subindices
To see how this works, imagine you have 3 documents: `doc1`, `doc2`, and `doc3`.

```python
doc1 = SimpleDirectoryReader('data1').load_data()
doc2 = SimpleDirectoryReader('data2').load_data()
doc3 = SimpleDirectoryReader('data3').load_data()
```

![](/_static/composability/diagram_b0.png)

Now let's define a tree index for each document. In Python, we have:

```python
index1 = GPTTreeIndex.from_documents(doc1)
index2 = GPTTreeIndex.from_documents(doc2)
index3 = GPTTreeIndex.from_documents(doc3)
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
    "What is a summary of this document?", mode="summarize"
)
index1_summary = str(summary)
```

**If specified**, this summary text for each subindex can be used to refine the answer during query-time. 

### Creating a Graph with a Top-Level Index

We can then create a graph with a list index on top of these 3 tree indices:
We can query, save, and load the graph to/from disk as any other index.

```python
from llama_index.indices.composability import ComposableGraph

graph = ComposableGraph.build_from_indices(
    GPTListIndex,
    [index1, index2, index3],
    index_summaries=[index1_summary, index2_summary, index3_summary],
)

# [Optional] save to disk
graph.save_to_disk("save_path.json")

# [Optional] load from disk
graph = ComposableGraph.load_from_disk("save_path.json")

```

![](/_static/composability/diagram.png)


### Querying the Graph

During a query, we would start with the top-level list index. Each node in the list corresponds to an underlying tree index. 
We want to make sure that we define a **recursive** query, as well as a **query config** list. If the query config list is not
provided, a default set will be used.
Information on how to specify query configs (either as a list of JSON dicts or `QueryConfig` objects) can be found 
[here](/reference/indices/composability_query.rst).


```python
# set query config. An example is provided below
query_configs = [
    {
        # NOTE: index_struct_id is optional
        "index_struct_id": "<index_id_1>",
        "index_struct_type": "tree",
        "query_mode": "default",
        "query_kwargs": {
            "child_branch_factor": 2
        }
    },
    {
        "index_struct_type": "keyword_table",
        "query_mode": "simple",
        "query_kwargs": {}
    },
    ...
]
response = graph.query("Where did the author grow up?", query_configs=query_configs)
```

> Note that specifying query config for index struct by id
> might require you to inspect e.g., `index1.index_struct.index_id`.
> Alternatively, you can explicitly set it as follows:
```python
index1.index_struct.index_id = "<index_id_1>"
index2.index_struct.index_id = "<index_id_2>"
index3.index_struct.index_id = "<index_id_3>"
```

![](/_static/composability/diagram_q1.png)

So within a node, instead of fetching the text, we would recursively query the stored tree index to retrieve our answer.

![](/_static/composability/diagram_q2.png)

NOTE: You can stack indices as many times as you want, depending on the hierarchies of your knowledge base! 


We can take a look at a code example below as well. We first build two tree indices, one over the Wikipedia NYC page, and the other over Paul Graham's essay. We then define a keyword extractor index over the two tree indices.

[Here is an example notebook](https://github.com/jerryjliu/gpt_index/blob/main/examples/composable_indices/ComposableIndices.ipynb).
