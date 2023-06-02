# Document Management

Most LlamaIndex index structures allow for **insertion**, **deletion**, **update**, and **refresh** operations.

## Insertion

You can "insert" a new Document into any index data structure, after building the index initially. This document will be broken down into nodes and ingested into the index.

The underlying mechanism behind insertion depends on the index structure. For instance, for the list index, a new Document is inserted as additional node(s) in the list.
For the vector store index, a new Document (and embeddings) is inserted into the underlying document/embedding store.

An example notebook showcasing our insert capabilities is given [here](https://github.com/jerryjliu/llama_index/blob/main/examples/paul_graham_essay/InsertDemo.ipynb).
In this notebook we showcase how to construct an empty index, manually create Document objects, and add those to our index data structures.

An example code snippet is given below:

```python
from llama_index import GPTListIndex, Document

index = GPTListIndex([])
text_chunks = ['text_chunk_1', 'text_chunk_2', 'text_chunk_3']

doc_chunks = []
for i, text in enumerate(text_chunks):
    doc = Document(text, doc_id=f"doc_id_{i}")
    doc_chunks.append(doc)

# insert
for doc_chunk in doc_chunks:
    index.insert(doc_chunk)
```

## Deletion

You can "delete" a Document from most index data structures by specifying a document_id. (**NOTE**: the tree index currently does not support deletion). All nodes corresponding to the document will be deleted.

```python
index.delete_ref_doc("doc_id_0", delete_from_docstore=True)
```

`delete_from_docstore` will default to `False` in case you are sharing nodes betweeen indexes using the same docstore. However, these nodes will not be used when querying when this is set to `False` as they will be deleted from the `index_struct` of the index, which keeps track of which nodes can be used for querying.

## Update

If a Document is already present within an index, you can "update" a Document with the same `doc_id` (for instance, if the information in the Document has changed).

```python
# NOTE: the document has a `doc_id` specified
doc_chunks[0].text = "Brand new document text"
index.update_ref_doc(
    doc_chunks[0], 
    update_kwargs={"delete_kwargs": {'delete_from_docstore': True}}
)
```

Here, we passed some extra kwargs to ensure the document is deleted from the docstore. This is of course optional.

## Refresh

If you set the `doc_id` of each document when loading your data, you can also automatically refresh the index.

The `refresh()` function will only update documents who have the same `doc_id`, but different text contents. Any documents not present in the index at all will also be inserted.

`refresh()` also returns a boolean list, indicating which documents in the input have been refreshed in the index.

```python
# modify first document, with the same doc_id
doc_chunks[0] = Document('Super new document text', doc_id="doc_id_0")

# add a new document
doc_chunks.append(Document("This isn't in the index yet, but it will be soon!", doc_id="doc_id_3"))

# refresh the index
refreshed_docs = index.refresh_ref_docs(
    doc_chunks,
    update_kwargs={"delete_kwargs": {'delete_from_docstore': True}}
)

# refreshed_docs[0] and refreshed_docs[-1] should be true
```

Again, we passed some extra kwargs to ensure the document is deleted from the docstore. This is of course optional.

If you `print()` the output of `refresh()`, you would see which input documents were refreshed:

```python
print(refreshed_docs)
> [True, False, False, True]
```

This is most useful when you are reading from a directory that is constantly updating with new information.

## Document Tracking

Any index that uses the docstore (i.e. all indexes except for most vector store integrations), you can also see which documents you have inserted into the docstore. 

```python
print(index.ref_doc_info)
> {'doc_id_1': RefDocInfo(doc_ids=['071a66a8-3c47-49ad-84fa-7010c6277479'], extra_info={}), 
   'doc_id_2': RefDocInfo(doc_ids=['9563e84b-f934-41c3-acfd-22e88492c869'], extra_info={}), 
   'doc_id_0': RefDocInfo(doc_ids=['b53e6c2f-16f7-4024-af4c-42890e945f36'], extra_info={}), 
   'doc_id_3': RefDocInfo(doc_ids=['6bedb29f-15db-4c7c-9885-7490e10aa33f'], extra_info={})}
```

Each entry in the output shows the ingested `doc_ids` as keys, and their associated `doc_ids` of the nodes they were split into. 

Lastly, the orignal `extra_info` dictionary of each input document is also tracked. You can read more about the `extra_info` attribute in [Customizing Documents](../customization/custom_documents.md).
