# Updating an Index

Every GPT Index data structure allows **insertion**, **deletion**, and **update**.

### Insertion

You can "insert" a new Document into any index data structure, after building the index initially. The underlying mechanism behind insertion depends on the index structure.
For instance, for the list index, a new Document is inserted as additional node(s) in the list.
For the vector store index, a new Document (and embedding) is inserted into the underlying document/embedding store.

An example notebook showcasing our insert capabilities is given [here](https://github.com/jerryjliu/gpt_index/blob/main/examples/paul_graham_essay/InsertDemo.ipynb).
In this notebook we showcase how to construct an empty index, manually create Document objects, and add those to our index data structures.

An example code snippet is given below:

```python
index = GPTListIndex([])

embed_model = OpenAIEmbedding()
doc_chunks = []
for i, text in enumerate(text_chunks):
    doc = Document(text, doc_id=f"doc_id_{i}")
    doc_chunks.append(doc)

# insert
for doc_chunk in doc_chunks:
    index.insert(doc_chunk)

```

### Deletion

You can "delete" a Document from most index data structures by specifying a document_id. (**NOTE**: the tree index currently does not support deletion). All nodes corresponding to
the document will be deleted.

**NOTE**: In order to delete a Document, that Document must have a doc_id specified when first loaded into the index.

```python
index.delete("doc_id_0")
```


### Update

If a Document is already present within an index, you can "update" a Document with the same `doc_id` (for instance, if the information in the Document has changed).

```python
# NOTE: the document has a `doc_id` specified
index.update(doc_chunks[0])
```





