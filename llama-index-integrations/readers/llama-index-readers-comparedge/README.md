# ComparEdge SaaS Reader for LlamaIndex

Loads SaaS product data from the [ComparEdge](https://comparedge.com) API into LlamaIndex Documents.

331 products, 28 categories. No auth.

```python
from llama_index.readers.comparedge import ComparEdgeReader

reader = ComparEdgeReader(category="crm")
docs = reader.load_data()
```

API: https://comparedge-api.up.railway.app/docs
