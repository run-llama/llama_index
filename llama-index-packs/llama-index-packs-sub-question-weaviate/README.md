# Sub Question Query Engine

This LlamaPack inserts your data into [Weaviate](https://weaviate.io/developers/weaviate) and uses the [Sub-Question Query Engine](https://gpt-index.readthedocs.io/en/latest/examples/query_engine/sub_question_query_engine.html) for your RAG application.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack WeaviateSubQuestionPack --download-dir ./weaviate_pack
```

You can then inspect the files at `./weaviate_pack` and use them as a template for your own project.

## Code Usage

You can download the pack to a the `./weaviate_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
WeaviateSubQuestionPack = download_llama_pack(
    "WeaviateSubQuestionPack", "./weaviate_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./weaviate_pack`.

Then, you can set up the pack like so:

```python
# setup pack arguments
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo

vector_store_info = VectorStoreInfo(
    content_info="brief biography of celebrities",
    metadata_info=[
        MetadataInfo(
            name="category",
            type="str",
            description=(
                "Category of the celebrity, one of [Sports Entertainment, Business, Music]"
            ),
        ),
    ],
)

import weaviate

client = weaviate.Client()

nodes = [...]

# create the pack
weaviate_pack = WeaviateSubQuestion(
    collection_name="test",
    vector_store_info=vector_store_index,
    nodes=nodes,
    client=client,
)
```

The `run()` function is a light wrapper around `query_engine.query()`.

```python
response = weaviate_pack.run("Tell me a bout a Music celebritiy.")
```

You can also use modules individually.

```python
# use the retriever
retriever = weaviate_pack.retriever
nodes = retriever.retrieve("query_str")

# use the query engine
query_engine = weaviate_pack.query_engine
response = query_engine.query("query_str")
```
