# LlamaIndex x LanceDB MultiModal AI LakeHouse

This package integrates the multi-modal functionalities of [LanceDB](https://lancedb.com) with LlamaIndex.

To install it, you can run:

```bash
pip install llama-index-indices-managed-lancedb
```

And you can then use it in your scripts as an index!

You can use it for text or images, and you can also employ it as a base for a retriever and a query engine.

## Text

You can use LanceDB with text in the following way:

```python
from llama_index.indices.managed.lancedb import LanceDBMultiModalIndex

# use it with a local database
local_index = LanceDBMultiModalIndex(
    uri="lancedb/data",
    text_embedding_model="sentence-transformers",
    embedding_model_kwargs={"name": "all-MiniLM-L6-v2"},
    table_name="documents",
)
# use a remote connection
remote_index = LanceDBMUltiModalIndex(
    uri="db://***",
    region="us-east-1",
    api_key="***",
    text_embedding_model="sentence-transformers",
    embedding_model_kwargs={"name": "all-MiniLM-L6-v2"},
    table_name="remote_documents",
)


# You always have to connect the index once you instantiated it with the primary constructor (__init__):
## 1. If you set use_async = True:
async def connect_lancedb_index():
    await documents_index.acreate_index()


## 2. If you set use_async = False (this is the default behavior):
local_index.create_index()

# load it from documents (async constructor)
from llama_index.core.schema import Document

document_data = [
    Document(text="This is an example document"),
    Document(text="This is as example document 1"),
]
documents_index = await LanceDBMUltiModalIndex.from_documents(
    documents=document_data,
    uri="lancedb/documents",
    text_embedding_model="sentence-transformers",
    embedding_model_kwargs={"name": "all-MiniLM-L6-v2"},
    table_name="from_documents",
    indexing="NO_INDEXING",
    use_async=True,
)
## load it from different type of data, e.g. PyArrow tables, Pandas/Polars DataFrames or list of dictionaries (async constructor)
import pandas as pd
import numpy as np

data = pd.DataFrame(
    {
        "text": ["## Hello world", "This is a test"],
        "id": ["1", "2"],
        "metadata": ['{"type": "text/markdown"}', '{"type": "text/plain"}'],
        "vector": [
            np.random.random(384).to_list(),
            np.random.random(384).to_list(),
        ],
    }
)
data_index = await LanceDBMUltiModalIndex.from_documents(
    documents=document_data,
    uri="lancedb/documents",
    text_embedding_model="sentence-transformers",
    embedding_model_kwargs={"name": "all-MiniLM-L6-v2"},
    table_name="from_data",
    indexing="HNSW_PQ",
    use_async=True,
)
```

We should notice three things here:

1. You can choose your own text embedding model among the ones [supported by LanceDB](https://lancedb.com/documentation/embeddings/index.html)
2. The schema for a text table is defined as followed:

```python
class TextSchema(LanceModel):
    id: str
    metadata: str  # deserializable
    text: str
    vector: List[List[float]]
```

In this schema, the text field is the source field for the embedding model to produce a vector, whereas the vector field must comply with the expected dimensions of the vectors produced by the embedding model. 3. You can define whether or not you want to index your table, and how to index it. Take a look at [LancDB docs](https://lancedb.com/documentation/guides/indexing/vector-index.html) to see what indexing strategies are available.

> [!IMPORTANT]
>
> In the following examples, we will be using only **sync** methods. It is nevertheless important to stress that, if you set `use_async = True`, you need to use the **async** corresponding methods.

Once you instantiated and connected the LanceDB index, you can:

**Add or delete nodes**

```python
local_index.insert_nodes(
    documents=[
        Document(text="Hello world", id_="1"),
        Document(text="How are you?", id_="2"),
    ],
)

# add from data
local_index.insert_data(
    data=pd.DataFrame(
        {
            "text": ["Hello world", "How are you?"],
            "id": ["1", "2"],
            "metadata": [
                '{"type": "text/markdown"}',
                '{"type": "text/plain"}',
            ],
        }
    ),
)

local_index.delete_nodes(["1", "2"])
```

**Retrieve**

```python
retriever = local_index.as_retriever()
nodes = retriever.retrieve(query_str="Hello world!")
print(nodes)
```

**Query**

```python
query_engine = local_index.as_query_engine()
response = query_engine.query(query_str="Hello world!")
print(response.response)
```

## Images

```python
images_index = LanceDBMultiModalIndex(
    uri="lancedb/images",
    multi_modal_embedding_model="open-clip",
    table_name="images",
)

# initialize from documents
from llama_index.core.schema import ImageDocument

images_index = await LanceDBMultiModalIndex.from_documents(
    documents=[
        ImageDocument(
            image_url="http://farm1.staticflickr.com/53/167798175_7c7845bbbd_z.jpg",
            metadata={"label": "cat"},
        ),
        ImageDocument(
            image_url="http://farm1.staticflickr.com/134/332220238_da527d8140_z.jpg",
            metadata={"label": "cat"},
        ),
        ImageDocument(
            image_url="http://farm9.staticflickr.com/8387/8602747737_2e5c2a45d4_z.jpg",
            metadata={"label": "dog"},
        ),
    ],
    uri="lancedb/images",
    multi_modal_embedding_model="open-clip",
    table_name="images",
)

# initialize from data
labels = ["dog", "horse", "horse"]
uris = [
    "http://farm5.staticflickr.com/4092/5017326486_1f46057f5f_z.jpg",
    "http://farm9.staticflickr.com/8216/8434969557_d37882c42d_z.jpg",
    "http://farm6.staticflickr.com/5142/5835678453_4f3a4edb45_z.jpg",
]
ids = [
    "1",
    "2",
    "3",
]
metadata = (
    [
        '{"mimetype": "image/jpeg"}',
        '{"mimetype": "image/jpeg"}',
        '{"mimetype": "image/jpeg"}',
    ],
)
image_bytes = [requests.get(uri).content for uri in uris]

data = pd.DataFrame(
    {
        "id": ids,
        "label": labels,
        "image_uri": uris,
        "image_bytes": image_bytes,
        "metadata": metadata,
    }
)

images_index = await LanceDBMultiModalIndex.from_data(
    data=data,
    uri="lancedb/images",
    multi_modal_embedding_model="open-clip",
    table_name="images",
)
```

As for before, you can choose your multi-modal embedding model and the index strategy, but this time the schema is a little bit different:

```python
class MultiModalSchema(LanceModel):
    id: str
    metadata: str  # deserializable
    label: str
    image_uri: str  # image uri as the source
    image_bytes: bytes  # image bytes as the source
    vector: List[List[float]]  # vector column
    vec_from_bytes: List[
        List[float]
    ]  # Another vector column (uses only bytes as source)
```

In this case, the source fields for the embedding model are `image_uri` and `image_bytes`.

You can use the index as for the text, but with a key difference in retrieving/querying: you use images!

```python
query_engine = images_index.as_query_engine()
# query_image can be a URL, an ImageBlock, an ImageDocument and a PIL Image
response = query_engine.query(
    query_image="http://farm6.staticflickr.com/5142/5835678453_4f3a4edb45_z.jpg"
)
# you can also use an image path
response = query_engine.query(
    query_image_path="/Users/user/images/hello_world.jpg"
)
```

## Extra features

1. You can initialize the index from an existing table, setting `table_exists = True` in the constructor methods.
2. There are methods (such as `insert` or `delete_ref_doc_id`) that work only for adding/deleting one node
3. If you set `use_async = True` you cannot use synchronous methods, and vice versa!
