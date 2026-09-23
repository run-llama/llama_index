---
title: Loading from LlamaParse
---

[LlamaParse](/llamaparse/), the hosted document platform from the LlamaIndex team (previously called LlamaCloud), can parse, index and query your data in a fully managed environment. Its [Index](/llamaparse/cloud-index-v2/getting_started/) product connects to your data sources, keeps the index in sync and serves retrieval, and the framework talks to it through `LlamaCloudIndex`. The class kept its name through the rename.

## Using LlamaParse from LlamaIndex

You can use Index to connect to your data stores and automatically index them. Once an index is created, you can use it in just a few lines of code:

```python
import os
from llama_cloud_services import LlamaCloudIndex

os.environ["LLAMA_CLOUD_API_KEY"] = "llx-..."

index = LlamaCloudIndex("my_first_index", project_name="Default")
query_engine = index.as_query_engine()
answer = query_engine.query("Example query")
```

It's also possible to load documents into an index programmatically; see the [Index documentation](/llamaparse/cloud-index-v2/getting_started/) and the [`LlamaCloudIndex` guide](/python/framework/module_guides/indexing/llama_cloud_index/) for details.
