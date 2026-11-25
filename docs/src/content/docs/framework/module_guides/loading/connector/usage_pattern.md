---
title: Usage Pattern
---

## Get Started

Readers are distributed as separate integration packages. Install the package for
the data source you need, then import its reader class directly into your
application.

Example usage:

```bash
pip install llama-index-readers-google
```

```python
from llama_index.core import VectorStoreIndex

from llama_index.readers.google import GoogleDocsReader

gdoc_ids = ["1wf-y2pd9C878Oh-FmLH7Q_BQkljdm6TQal-c1pUfrec"]
loader = GoogleDocsReader()
documents = loader.load_data(document_ids=gdoc_ids)
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
query_engine.query("Where did the author go to school?")
```
