# Composability


GPT Index offers **composability** of your indices, meaning that you can build indices on top of other indices. This allows you to more effectively index your entire document tree in order to feed custom knowledge to GPT.

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
index1 = GPTTreeIndex(doc1)
index2 = GPTTreeIndex(doc2)
index3 = GPTTreeIndex(doc3)
```

![](/_static/composability/diagram_b1.png)

### Defining Summary Text

You then need to explicitly define *summary text* for each subindex. This allows  
the subindices to be used as Documents for higher-level indices.

```python
index1.set_text("<summary1>")
index2.set_text("<summary2>")
index3.set_text("<summary3>")
```

You may choose to manually specify the summary text, or use GPT Index itself to generate
a summary, for instance with the following:

```python
summary = index1.query(
    "What is a summary of this document?", mode="summarize"
)
index1.set_text(str(summary))
```

**If specified**, this summary text for each subindex can be used to refine the answer during query-time. 

### Defining a Top-Level Index

We can then create a list index on these 3 tree indices:

```python
list_index = GPTListIndex([index1, index2, index3])
```

![](/_static/composability/diagram.png)


### Defining a Graph Structure


Finally, we define a `ComposableGraph` to "wrap" the composed index graph.
We can do this by simply feeding in the top-level index.
This wrapper allows us to query, save, and load the graph to/from disk.

```python

from gpt_index.composability import ComposableGraph

graph = ComposableGraph.build_from_index(list_index)

# [Optional] save to disk
graph.save_to_disk("save_path.json")

# [Optional] load from disk
graph = ComposableGraph.load_from_disk("save_path.json")

```


### Querying the Top-Level Index

During a query, we would start with the top-level list index. Each node in the list corresponds to an underlying tree index. 
We want to make sure that we define a **recursive** query, as well as a **query config** list. If the query config list is not
provided, a default set will be used.
Information on how to specify query configs (either as a list of JSON dicts or `QueryConfig` objects) can be found 
[here](/reference/indices/composability_query.rst).


```python
# set query config. An example is provided below
query_configs = [
    {
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

![](/_static/composability/diagram_q1.png)

So within a node, instead of fetching the text, we would recursively query the stored tree index to retrieve our answer.

![](/_static/composability/diagram_q2.png)

NOTE: You can stack indices as many times as you want, depending on the hierarchies of your knowledge base! 


We can take a look at a code example below as well. We first build two tree indices, one over the Wikipedia NYC page, and the other over Paul Graham's essay. We then define a keyword extractor index over the two tree indices.

[Here is an example notebook](https://github.com/jerryjliu/gpt_index/blob/main/examples/composable_indices/ComposableIndices.ipynb).
