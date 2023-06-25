# Customizing Documents

This section covers various ways to customize `Document` objects. Since the `Document` object is a subclass of our `TextNode` object, all these settings and details apply to the `TextNode` object class as well.

## Metadata

Documents also offer the chance to include useful metadata. Using the `metadata` dictionary on each document, additional information can be included to help inform responses and track down sources for query responses. This information can be anything, such as filenames or categories. If you are intergrating with a vector database, keep in mind that some vector databases require that the keys must be strings, and the values must be flat (either `str`, `float`, or `int`).

Any information set in the `metadata` dictionary of each document will show up in the `metadata` of each source node created from the document. Additionaly, this information is included in the nodes, enabling the index to utilize it on queries and responses. By default, the metadata is injected into the text for both embedding and LLM model calls.

There are a few ways to set up this dictionary:

1. In the document constructor:

```python
document = Document(
    text='text', 
    metadata={
        'filename': '<doc_file_name>', 
        'category': '<category>'
    }
)
```

2. After the document is created:

```python
document.metadata = {'filename': '<doc_file_name>'}
```

3. Set the filename automatically using the `SimpleDirectoryReader` and `file_metadata` hook. This will automatically run the hook on each document to set the `metadata` field:

```python
from llama_index import SimpleDirectoryReader
filename_fn = lambda filename: {'file_name': filename}

# automatically sets the metadata of each document according to filename_fn
documents = SimpleDirectoryReader('./data', file_metadata=filename_fn)
```

## Customizing the id

As detailed in the section [Document Management](../index/document_management.md), the doc `id_` is used to enable effecient refreshing of documents in the index. When using the `SimpleDirectoryReader`, you can automatically set the doc `id_` to be the full path to each document:

```python
from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data", filename_as_id=True).load_data()
print([x.doc_id for x in documents])
```

You can also set the `id_` of any `Document` or `TextNode` directly!

```python
document.id_ = "My new document id!"
```

## Advanced - Metadata Customization

A key detail mentioned above is that by default, any metadata you set is included in the embeddings generation and LLM.

### Customizing LLM Metadata Text

Typically, a document might have many metadata keys, but you might not want all of them visibile to the LLM during response synthesis. In the above examples, we may not want the LLM to read the `file_name` of our document. However, the `file_name` might include information that will help generate better embeddings. A key advantage of doing this is to bias the embeddings for retrieval without changing what the LLM ends up reading. 

We can exclude it like so:

```python
document.excluded_llm_metadata_keys = ['file_name']
```

Then, we can test what the LLM will actually end up reading using the `get_content()` function and specifying `MetadataMode.LLM`:

```python
from llama_index.schema import MetadataMode
print(document.get_content(metadata_mode=MetadataMode.LLM))
```

### Customizing Embedding Metadata Text

Similar to customing the metadata visibile to the LLM, we can also customize the metadata visible to emebddings. In this case, you can specifically exclude metadata visible to the embedding model, in case you DON'T want particular text to bias the embeddings.

```python
document.excluded_embed_metadata_keys = ['file_name']
```

Then, we can test what the embedding model will actually end up reading using the `get_content()` function and specifying `MetadataMode.EMBED`:

```python
from llama_index.schema import MetadataMode
print(document.get_content(metadata_mode=MetadataMode.EMBED))
```

### Customizing Metadata Format

As you know by now, metadata is injected into the actual text of each document/node when sent to the LLM or embedding model. By default, the format of this metadata is controlled by three attributes:

1. `Document.metadata_seperator` -> default = `"\n"`

When concatenating all key/value fields of your metadata, this field controls the seperator bewtween each key/value pair.

2. `Document.metadata_template` -> default = `"{key}: {value}"`

This attribute controls how each key/value pair in your metadata is formatted. The two variables `key` and `value` string keys are required.

3. `Document.text_template` -> default = `{metadata_str}\n\n{content}`

Once your metadata is converted into a string using `metadata_seperator` and `metadata_template`, this templates controls what that metadata looks like when joined with the text content of your document/node. The `metadata` and `content` string keys are required.

### Summary

Knowing all this, let's create a short example using all this power:

```python
from llama_index import Document
from llama_index.schema import MetadataMode

document = Document(
    text="This is a super-customized document",
    metadata={
        "file_name": "super_secret_document.txt",
        "category": "finance",
        "author": "LlamaIndex"    
    },
    excluded_llm_metadata_keys=['file_name'],
    metadata_seperator="::",
    metadata_template="{key}=>{value}",
    text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
)

print("The LLM sees this: \n", document.get_content(metadata_mode=MetadataMode.LLM))
print("The Embedding model sees this: \n", document.get_content(metadata_mode=MetadataMode.EMBED))
```

## Advanced - Document Weighting

Some documents are more important than others. If you are using the default vector store in LlamaIndex, you can take advantage of weighted retrieval! Each `Document` can be assigned a weight (which is also inherited to it's nodes) that used used during vector retrieval. The weight is simply multiplied to the base similarity score when calculating similarity between text and the query string to an index.

```python
document = Document(text="Super important text", weight=1.2)
```

In this case, if the base similarity score was 0.8, this weight will increase it to 0.96, making it more likely to be retrieved and seen by the LLM!
