## Usage Pattern

Each data loader contains a "Usage" section showing how that loader can be used. At the core of using each loader is a `download_loader` function, which
downloads the loader file into a module that you can use within your application.

Example usage:

```python
from llama_index import GPTVectorStoreIndex, download_loader

GoogleDocsReader = download_loader('GoogleDocsReader')

gdoc_ids = ['1wf-y2pd9C878Oh-FmLH7Q_BQkljdm6TQal-c1pUfrec']
loader = GoogleDocsReader()
documents = loader.load_data(document_ids=gdoc_ids)
index = GPTVectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
query_engine.query('Where did the author go to school?')
```
