# Timescale Vector AutoRetrieval Pack With Hybrid Search on Time

This pack demonstrates performing auto-retrieval for hybrid search based on both similarity and time, using the timescale-vector (PostgreSQL) vectorstore.
This sort of time-based retrieval is particularly effective for data where time is a key element of the data, such as:

- News articles (covering various timely topics like politics or business)
- Blog posts, documentation, or other published materials (both public and private)
- Social media posts
- Changelogs
- Messages
- Any timestamped data

Hybrid search of similarity and time method is ideal for queries that require sorting by semantic relevance while filtering by time and date.
For instance: (1) Finding recent posts about NVDA in the past 7 days (2) Finding all news articles related to music celebrities from 2020.

[Timescale Vector](https://www.timescale.com/ai?utm_campaign=vectorlaunch&utm_source=llamaindex&utm_medium=referral) is a PostgreSQL-based vector database that provides superior performance when searching for embeddings within a particular timeframe by leveraging automatic table partitioning to isolate data for particular time-ranges.

The auto-retriever will use the LLM at runtime to set metadata filters (including deducing the time-ranges to search), a top-k value, and the query string for similarity search based on the text of user queries. That query will then be executed on the vector store.

## What is Timescale Vector?

**[Timescale Vector](https://www.timescale.com/ai?utm_campaign=vectorlaunch&utm_source=llamaindex&utm_medium=referral) is PostgreSQL++ for AI applications.**

Timescale Vector enables you to efficiently store and query millions of vector embeddings in `PostgreSQL`.

- Enhances `pgvector` with faster and more accurate similarity search on millions vectors via a DiskANN inspired indexing algorithm.
- Enables fast time-based vector search via automatic time-based partitioning and indexing.

Timescale Vector is cloud PostgreSQL for AI that scales with you from POC to production:

- Simplifies operations by enabling you to store relational metadata, vector embeddings, and time-series data in a single database.
- Benefits from rock-solid PostgreSQL foundation with enterprise-grade feature liked streaming backups and replication, high-availability and row-level security.
- Enables a worry-free experience with enterprise-grade security and compliance.

### How to access Timescale Vector

Llama index users get a 90-day free trial for Timescale Vector. [Sign up here](https://console.cloud.timescale.com/signup?utm_campaign=vectorlaunch&utm_source=llamaindex&utm_medium=referral) for a free cloud vector database.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack TimescaleVectorAutoretrievalPack --download-dir ./tsv_pack
```

You can then inspect the files at `./tsv_pack` and use them as a template for your own project.

## Code Usage

You can download the pack to a the `./tsv_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
TimescaleVectorAutoretrievalPack = download_llama_pack(
    "TimescaleVectorAutoretrievalPack", "./tsv_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./tsv_pack`.

Then, you can set up the pack like so:

```python
# setup pack arguments
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from timescale_vector import client
from dotenv import load_dotenv, find_dotenv
import os
from datetime import timedelta

# this is an example of the metadata describing the nodes. The example is for git commit history.
vector_store_info = VectorStoreInfo(
    content_info="Description of the commits to PostgreSQL. Describes changes made to Postgres",
    metadata_info=[
        MetadataInfo(
            name="commit_hash",
            type="str",
            description="Commit Hash",
        ),
        MetadataInfo(
            name="author",
            type="str",
            description="Author of the commit",
        ),
        # "__start_date" is a special reserved name referring to the starting point for the time of the uuid field.
        MetadataInfo(
            name="__start_date",
            type="datetime in iso format",
            description="All results will be after this datetime",
        ),
        # "__end_date" is a special reserved name referring to the last point for the time of the uuid field.
        MetadataInfo(
            name="__end_date",
            type="datetime in iso format",
            description="All results will be before this datetime",
        ),
    ],
)

# nodes have to have their `id_` field set using a V1 UUID with the right time component
# this can be achieved by using `client.uuid_from_time(datetime_obj)`
nodes = [...]
# an example:
# nodes = [
#    TextNode(
#        id_=str(client.uuid_from_time(datetime(2021, 1, 1))),
#        text="My very interesting commit message",
#        metadata={
#            "author": "Matvey Arye",
#        },
#    )
# ]

_ = load_dotenv(find_dotenv(), override=True)
service_url = os.environ["TIMESCALE_SERVICE_URL"]

# choose a time_partition_interval for your data
# the interval should be chosen so that most queries
# will touch 1-2 partitions while all your data should
# fit in less than 1000 partitions.
time_partition_interval = timedelta(days=30)

# create the pack
tsv_pack = TimescaleVectorAutoretrievalPack(
    service_url=service_url,
    table_name="test",
    time_partition_interval=time_partition_interval,
    vector_store_info=vector_store_info,
    nodes=nodes,
)
```

The `run()` function is a light wrapper around `query_engine.query()`.

```python
response = tsv_pack.run(
    "What new features were added in the past three months?"
)
```

You can also use modules individually.

```python
# use the retriever
retriever = tsv_pack.retriever
nodes = retriever.retrieve("query_str")

# use the query engine
query_engine = tsv_pack.query_engine
response = query_engine.query("query_str")
```
