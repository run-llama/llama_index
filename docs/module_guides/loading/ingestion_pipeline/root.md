# Ingestion Pipeline

An `IngestionPipeline` uses a concept of `Transformations` that are applied to input data. These `Transformations` are applied to your input data, and the resulting nodes are either returned or inserted into a vector database (if given). Each node+transformation pair is cached, so that subsequent runs (if the cache is persisted) with the same node+transformation combination can use the cached result and save you time.

## Usage Pattern

At it's most basic level, you can quickly instantiate an `IngestionPipeline` like so:

```python
from llama_index import Document
from llama_index.embeddings import OpenAIEmbedding
from llama_index.text_splitter import SentenceSplitter
from llama_index.extractors import TitleExtractor
from llama_index.ingestion import IngestionPipeline, IngestionCache

# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
        OpenAIEmbedding(),
    ]
)

# run the pipeline
nodes = pipeline.run(documents=[Document.example()])
```

## Connecting to Vector Databases

When running an ingestion pipeline, you can also chose to automatically insert the resulting nodes into a remote vector store.

Then, you can construct an index from that vector store later on.

```python
from llama_index import Document
from llama_index.embeddings import OpenAIEmbedding
from llama_index.text_splitter import SentenceSplitter
from llama_index.extractors import TitleExtractor
from llama_index.ingestion import IngestionPipeline
from llama_index.vector_stores.qdrant import QdrantVectorStore

import qdrant_client

client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="test_store")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
        OpenAIEmbedding(),
    ],
    vector_store=vector_store,
)

# Ingest directly into a vector db
pipeline.run(documents=[Document.example()])

# Create your index
from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_vector_store(vector_store)
```

## Caching

In an `IngestionPipeline`, each node + transformation combination is hashed and cached. This saves time on subsequent runs that use the same data.

The following sections describe some basic usage around caching.

### Local Cache Management

Once you have a pipeline, you may want to store and load the cache.

```python
# save and load
pipeline.cache.persist("./test_cache.json")
new_cache = IngestionCache.from_persist_path("./test_cache.json")

new_pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
    ],
    cache=new_cache,
)

# will run instantly due to the cache
nodes = pipeline.run(documents=[Document.example()])
```

If the cache becomes too large, you can clear it

```python
# delete all context of the cache
cache.clear()
```

### Remote Cache Management

We support multiple remote storage backends for caches

- `RedisCache`
- `MongoDBCache`
- `FirestoreCache`

Here as an example using the `RedisCache`:

```python
from llama_index import Document
from llama_index.embeddings import OpenAIEmbedding
from llama_index.text_splitter import SentenceSplitter
from llama_index.extractors import TitleExtractor
from llama_index.ingestion import IngestionPipeline, IngestionCache
from llama_index.ingestion.cache import RedisCache


pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
        OpenAIEmbedding(),
    ],
    cache=IngestionCache(
        cache=RedisCache(
            redis_uri="redis://127.0.0.1:6379", collection="test_cache"
        )
    ),
)

# Ingest directly into a vector db
nodes = pipeline.run(documents=[Document.example()])
```

Here, no persist step is needed, since everything is cached as you go in the specified remote collection.

## Async Support

The `IngestionPipeline` also has support for async operation

```python
nodes = await pipeline.arun(documents=documents)
```

## Modules

```{toctree}
---
maxdepth: 2
---
transformations.md
```
