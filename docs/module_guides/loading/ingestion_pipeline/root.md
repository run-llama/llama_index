# Ingestion Pipeline

An `IngestionPipeline` uses a concept of `Transformations` that are applied to input data. These `Transformations` are applied to your input data, and the resulting nodes are either returned or inserted into a vector database (if given). Each node+transformation pair is cached, so that subsequent runs (if the cache is persisted) with the same node+transformation combination can use the cached result and save you time.

## Usage Pattern

The simplest usage is to instantiate an `IngestionPipeline` like so:

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

Note that in a real-world scenario, you would get your documents from `SimpleDirectoryReader` or another reader from Llama Hub.

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

## Calculating embeddings in a pipeline

Note that in the above example, embeddings are calculated as part of the pipeline. If you are connecting your pipeline to a vector store, embeddings must be a stage of your pipeline or your later instantiation of the index will fail.

You can omit embeddings from your pipeline if you are not connecting to a vector store, i.e. just producing a list of nodes.

## Caching

In an `IngestionPipeline`, each node + transformation combination is hashed and cached. This saves time on subsequent runs that use the same data.

The following sections describe some basic usage around caching.

### Local Cache Management

Once you have a pipeline, you may want to store and load the cache.

```python
# save
pipeline.persist("./pipeline_storage")

# load and restore state
new_pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
    ],
)
new_pipeline.load("./pipeline_storage")

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

## Document Management

Attaching a `docstore` to the ingestion pipeline will enable document management.

Using the `document.doc_id` or `node.ref_doc_id` as a grounding point, the ingestion pipeline will actively look for duplicate documents.

It works by:

- Storing a map of `doc_id` -> `document_hash`
- If a vector store is attached:
  - If a duplicate `doc_id` is detected, and the hash has changed, the document will be re-processed and upserted
  - If a duplicate `doc_id` is detected and the hash is unchanged, the node is skipped
- If only a vector store is not attached:
  - Checks all existing hashes for each node
  - If a duplicate is found, the node is skipped
  - Otherwise, the node is processed

**NOTE:** If we do not attach a vector store, we can only check for and remove duplicate inputs.

```python
from llama_index.ingestion import IngestionPipeline
from llama_index.storage.docstore import SimpleDocumentStore

pipeline = IngestionPipeline(
    transformations=[...], docstore=SimpleDocumentStore()
)
```

A full walkthrough is found in our [demo notebook](/examples/ingestion/document_management_pipeline.ipynb).

Also check out another guide using [Redis as our entire ingestion stack](/examples/ingestion/redis_ingestion_pipeline.ipynb).

## Modules

```{toctree}
---
maxdepth: 2
---
transformations.md
/examples/ingestion/advanced_ingestion_pipeline.ipynb
/examples/ingestion/async_ingestion_pipeline.ipynb
/examples/ingestion/document_management_pipeline.ipynb
/examples/ingestion/redis_ingestion_pipeline.ipynb
/examples/ingestion/ingestion_gdrive.ipynb
```
