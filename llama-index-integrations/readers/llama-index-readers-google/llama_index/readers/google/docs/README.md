# Google Doc Loader

`pip install llama-index-readers-google`

This loader takes in IDs of Google Docs and parses their text into `Document`s. You can extract a Google Doc's ID directly from its URL. For example, the ID of `https://docs.google.com/document/d/1wf-y2pd9C878Oh-FmLH7Q_BQkljdm6TQal-c1pUfrec/edit` is `1wf-y2pd9C878Oh-FmLH7Q_BQkljdm6TQal-c1pUfrec`.

As a prerequisite, you will need to register with Google and generate a `credentials.json` file in the directory where you run this loader. See [here](https://developers.google.com/workspace/guides/create-credentials) for instructions.

## Usage

To use this loader, you simply need to pass in an array of Google Doc IDs.

```python
from llama_index.readers.google import GoogleDocsReader

gdoc_ids = ["1wf-y2pd9C878Oh-FmLH7Q_BQkljdm6TQal-c1pUfrec"]
loader = GoogleDocsReader()
documents = loader.load_data(document_ids=gdoc_ids)
```

## Examples

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).

### LlamaIndex

```python
from llama_index.readers.google import GoogleDocsReader
from llama_index.core import VectorStoreIndex, download_loader

gdoc_ids = ["1wf-y2pd9C878Oh-FmLH7Q_BQkljdm6TQal-c1pUfrec"]
loader = GoogleDocsReader()
documents = loader.load_data(document_ids=gdoc_ids)
index = VectorStoreIndex.from_documents(documents)
index.query("Where did the author go to school?")
```
