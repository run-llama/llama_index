# Fleet Context Embeddings - Building a Hybrid Search Engine for the Llamaindex Library

In this guide, we will be using Fleet Context to download the embeddings for LlamaIndex's documentation and build a hybrid dense/sparse vector retrieval engine on top of it.

<br><br>

## Pre-requisites

```
!pip install llama-index
!pip install --upgrade fleet-context
```

```
import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..." # add your API key here!
openai.api_key = os.environ["OPENAI_API_KEY"]
```

<br><br>

## Download Embeddings from Fleet Context

We will be using Fleet Context to download the embeddings for the
entirety of LlamaIndex\'s documentation (\~12k chunks, \~100mb of
content). You can download for any of the top 1220 libraries by
specifying the library name as a parameter. You can view the full list
of supported libraries [here](https://fleet.so/context) at the bottom of
the page.

We do this because Fleet has built a embeddings pipeline that preserves
a lot of important information that will make the retrieval and
generation better including position on page (for re-ranking), chunk
type (class/function/attribute/etc), the parent section, and more. You
can read more about this on their [Github
page](https://github.com/fleet-ai/context/tree/main).

```python
from context import download_embeddings

df = download_embeddings("llamaindex")
```

**Output**:

```shell
    100%|██████████| 83.7M/83.7M [00:03<00:00, 27.4MiB/s]
                                         id  \
    0  e268e2a1-9193-4e7b-bb9b-7a4cb88fc735
    1  e495514b-1378-4696-aaf9-44af948de1a1
    2  e804f616-7db0-4455-9a06-49dd275f3139
    3  eb85c854-78f1-4116-ae08-53b2a2a9fa41
    4  edfc116e-cf58-4118-bad4-c4bc0ca1495e
```

```python
# Show some examples of the metadata
df["metadata"][0]
display(Markdown(f"{df['metadata'][8000]['text']}"))
```

**Output**:

```shell
classmethod from_dict(data: Dict[str, Any], kwargs: Any) → Self classmethod from_json(data_str: str, kwargs: Any) → Self classmethod from_orm(obj: Any) → Model json(, include: Optional[Union[AbstractSetIntStr, MappingIntStrAny]] = None, exclude: Optional[Union[AbstractSetIntStr, MappingIntStrAny]] = None, by_alias: bool = False, skip_defaults: Optional[bool] = None, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False, encoder: Optional[Callable[[Any], Any]] = None, models_as_dict: bool = True*, dumps_kwargs: Any) → unicode Generate a JSON representation of the model, include and exclude arguments as per dict().
```

<br><br>

## Create Pinecone Index for Hybrid Search in LlamaIndex

We\'re going to create a Pinecone index and upsert our vectors there so
that we can do hybrid retrieval with both sparse vectors and dense
vectors. Make sure you have a [Pinecone account](https://pinecone.io)
before you proceed.

```python
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
```

```python
import pinecone

api_key = "..."  # Add your Pinecone API key here
pinecone.init(
    api_key=api_key, environment="us-east-1-aws"
)  # Add your db region here
```

```python
# Fleet Context uses the text-embedding-ada-002 model from OpenAI with 1536 dimensions.

# NOTE: Pinecone requires dotproduct similarity for hybrid search
pinecone.create_index(
    "quickstart-fleet-context",
    dimension=1536,
    metric="dotproduct",
    pod_type="p1",
)

pinecone.describe_index(
    "quickstart-fleet-context"
)  # Make sure you create an index in pinecone
```

<br>

```python
from llama_index.vector_stores.pinecone import PineconeVectorStore

pinecone_index = pinecone.Index("quickstart-fleet-context")
vector_store = PineconeVectorStore(pinecone_index, add_sparse_vector=True)
```

<br><br>

## Batch upsert vectors into Pinecone

Pinecone recommends upserting 100 vectors at a time. We\'re going to do that after we modify the format of the data a bit.

```python
import random
import itertools


def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


# generator that generates many (id, vector, metadata, sparse_values) pairs
data_generator = map(
    lambda row: {
        "id": row[1]["id"],
        "values": row[1]["values"],
        "metadata": row[1]["metadata"],
        "sparse_values": row[1]["sparse_values"],
    },
    df.iterrows(),
)

# Upsert data with 1000 vectors per upsert request
for ids_vectors_chunk in chunks(data_generator, batch_size=100):
    print(f"Upserting {len(ids_vectors_chunk)} vectors...")
    pinecone_index.upsert(vectors=ids_vectors_chunk)
```

<br><br>

## Build Pinecone Vector Store in LlamaIndex

Finally, we\'re going to build the Pinecone vector store via LlamaIndex
and query it to get results.

```python
from llama_index.core import VectorStoreIndex
from IPython.display import Markdown, display
```

```python
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
```

<br><br>

## Query Your Index!

```python
query_engine = index.as_query_engine(
    vector_store_query_mode="hybrid", similarity_top_k=8
)
response = query_engine.query("How do I use llama_index SimpleDirectoryReader")
```

```python
display(Markdown(f"<b>{response}</b>"))
```

**Output**:

```shell
<b>To use the SimpleDirectoryReader in llama_index, you need to import it from the llama_index library. Once imported, you can create an instance of the SimpleDirectoryReader class by providing the directory path as an argument. Then, you can use the `load_data()` method on the SimpleDirectoryReader instance to load the documents from the specified directory.</b>
```
