# Google Keep Loader

`pip install llama-index-readers-google`

This loader takes in IDs of Google Keep and parses their text into `Document`s. You can extract a Google Keep's ID directly from its URL. For example, the ID of `https://keep.google.com/u/6/#NOTE/1OySsaIrx_pvQaJJk3VPQfYQvSuxTQuPndEEGl7qvrhFaN8VnO4K8Bti0SL2YklU` is `1OySsaIrx_pvQaJJk3VPQfYQvSuxTQuPndEEGl7qvrhFaN8VnO4K8Bti0SL2YklU`.

This loader uses the (unofficial) gkeepapi library. Google Keep does provide an official API, however in order to use it, (1) your account has to be an Enterprise (Google Workspace) account (2) you will need to generate a service account to authenticate with Google Keep API (3) you will need to enable Domain-wide Delegation to enable the service account with Google Read API scopes. See [here](https://issuetracker.google.com/issues/210500028) for details. Thus I believe gkeepapi is actually more practical and useful for the majority of the users.

To use gkeepapi, you will need to login with username and a password. I highly recommend using a (one-off) App Password over using your own password. You can find how to generate App Password at [here](https://support.google.com/accounts/answer/185833?hl=en). The username and password should be saved at a `keep_credentials.json` file, with `username` and `password` being keys. It's recommended you delete the App Password once you no longer need it.

## Usage

To use this loader, you simply need to pass in an array of Google Keep IDs.

```python
from llama_index.readers.google import GoogleKeepReader

gkeep_ids = ["1wf-y2pd9C878Oh-FmLH7Q_BQkljdm6TQal-c1pUfrec"]
loader = GoogleKeepReader()
documents = loader.load_data(document_ids=gkeep_ids)
```

## Examples

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).

### LlamaIndex

```python
from llama_index import VectorStoreIndex
from llama_index.readers.google import GoogleKeepReader

gkeep_ids = ["1wf-y2pd9C878Oh-FmLH7Q_BQkljdm6TQal-c1pUfrec"]
loader = GoogleKeepReader()
notes = loader.load_data(document_ids=gkeep_ids)
index = VectorStoreIndex.from_documents(notes)
query_engine = index.as_query_engine()
query_engine.query("What are my current TODOs?")
```
