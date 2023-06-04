# Usage Pattern


## Get Started
Get a retriever from index:
```python
retriever = index.as_retriever()
```

Retrieve relevant context for a question:
```python
nodes = retriever.retrieve('Who is Paul Graham?')
```

> Note: To learn how to build an index, see [Index](/how_to/index/root.md)


## Advanced
TODO