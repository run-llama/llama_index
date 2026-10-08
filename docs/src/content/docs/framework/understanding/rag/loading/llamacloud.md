---
title: Loading from LlamaParse
---

[LlamaParse](/llamaparse/), the hosted document platform from the LlamaIndex team (previously called LlamaCloud), can parse, index and query your data in a fully managed environment. Its Index product connects to your data sources, keeps the index in sync and serves retrieval, and the framework talks to the first version of it through `LlamaCloudIndex`.

:::caution[This integration targets the earlier version of Index]
`LlamaCloudIndex` ships in `llama-cloud-services`, a package that is deprecated and no longer updated, and it talks to the first version of Index. Installing it pins an old release of `llama-cloud`, so it cannot share an environment with the current `llama-cloud` SDK. For new projects, use [Index v2](/llamaparse/cloud-index-v2/getting_started/) through the `llama-cloud` SDK.
:::

## Using LlamaParse from LlamaIndex

You can use Index to connect to your data stores and automatically index them. Once an index is created, you can use it in just a few lines of code:

```bash
pip install llama-index llama-cloud-services
```

```python
import os
from llama_cloud_services import LlamaCloudIndex

os.environ["LLAMA_CLOUD_API_KEY"] = "llx-..."

index = LlamaCloudIndex("my_first_index", project_name="Default")
query_engine = index.as_query_engine()
answer = query_engine.query("Example query")
```

It's also possible to load documents into an index programmatically; see the [documentation for that version of Index](/llamaparse/deprecated/cloud-index/getting_started/) and the [`LlamaCloudIndex` guide](/python/framework/module_guides/indexing/llama_cloud_index/) for details.
