---
title: Usage Pattern
---

## Get Started

Each data loader contains a "Usage" section showing how that loader can be used. Install the relevant reader package and import the reader class directly in your application.

Example usage:

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
