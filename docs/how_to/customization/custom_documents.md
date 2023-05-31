### Customizing Documents

Documents also offer the chance to include useful metadata. Using the `extra_info` dictionary on each document, additional information can be included to help inform responses and track down sources for query responses. This information can be anything, such as filenames or categories, but the only requirement is that the keys must be strings, and the values must be either `str`, `float`, or `int`.

Any information set in the `extra_info` dictionary of each document will show up in the `extra_info` of each source node created from the document.

There are a few ways to set up this dictionary:

1. In the document constructor:

```python
document = Document(
    'text', 
    extra_info={
        'filename', '<doc_file_name>', 
        'category': '<category>'
    }
)
```

2. After the document is created:

```python
document.extra_info = {'filename', '<doc_file_name>'}
```

3. Set the filename automatically using the `SimpleDirectoryReader` and `file_metadata` hook. This will automatically run the hook on each document to set the `extra_info` field:

```python
from llama_index import SimpleDirectoryReader
filename_fn = lambda filename: {'file_name': filename}

# automatically sets the extra_info of each document according to filename_fn
documents = SimpleDirectoryReader('./data', file_metadata=filename_fn)
```
