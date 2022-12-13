# Composability


GPT Index offers **composability** of your indices, meaning that you can build indices on top of other indices. This allows you to more effectively index your entire document tree in order to feed custom knowledge to GPT.

Composability allows you to to define lower-level indices for each document, and higher-order indices over a collection of documents. To see how this works, imagine defining 1) a tree index for the text within each document, and 2) a list index over each tree index (one document) within your collection.

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
index2 = GPTTreeIndex(doc2)
```

![](/_static/composability/diagram_b1.png)

We can then create a list index on these 3 tree indices:

```python
list_index = GPTListIndex([index1, index2, index3])
```

![](/_static/composability/diagram.png)

During a query, we would start with the top-level list index. Each node in the list corresponds to an underlying tree index. 

```python
response = list_index.query("Where did the author grow up?")
```

![](/_static/composability/diagram_q1.png)

So within a node, instead of fetching the text, we would recursively query the stored tree index to retrieve our answer.

![](/_static/composability/diagram_q2.png)

NOTE: You can stack indices as many times as you want, depending on the hierarchies of your knowledge base! 


We can take a look at a code example below as well. We first build two tree indices, one over the Wikipedia NYC page, and the other over Paul Graham's essay. We then define a keyword extractor index over the two tree indices.

[Here is an example notebook](https://github.com/jerryjliu/gpt_index/blob/main/examples/composable_indices/ComposableIndices.ipynb).
