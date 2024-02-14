# Readers (Loaders)

## Reader Usage (Use `download_loader` from LlamaIndex)

You can also use the loaders with `download_loader` from LlamaIndex in a single line of code.

For example, see the code snippets below using the Google Docs Loader.

```python
from llama_index.core import VectorStoreIndex, download_loader

GoogleDocsReader = download_loader("GoogleDocsReader")

gdoc_ids = ["1wf-y2pd9C878Oh-FmLH7Q_BQkljdm6TQal-c1pUfrec"]
loader = GoogleDocsReader()
documents = loader.load_data(document_ids=gdoc_ids)
index = VectorStoreIndex.from_documents(documents)
index.query("Where did the author go to school?")
```
