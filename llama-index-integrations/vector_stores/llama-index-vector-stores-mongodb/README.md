# LlamaIndex Vector_Stores Integration: MongoDB

## Setting up MongoDB Atlas as the Datastore Provider

MongoDB Atlas is a multi-cloud database service made by the same people that build MongoDB.
Atlas simplifies deploying and managing your databases while offering the versatility you need
to build resilient and performant global applications on the cloud providers of your choice.

You can perform semantic search on data in your Atlas cluster running MongoDB v6.0.11, v7.0.2,
or later using Atlas Vector Search. You can store vector embeddings for any kind of data along
with other data in your collection on the Atlas cluster.

In the section, we provide detailed instructions to run the tests.

### Deploy a Cluster

Follow the [Getting-Started](https://www.mongodb.com/basics/mongodb-atlas-tutorial) documentation
to create an account, deploy an Atlas cluster, and connect to a database.

### Retrieve the URI used by Python to connect to the Cluster

Once deployed, you will need a URI (connection string) to access the cluster.
This you should store as the environment variable: `MONGODB_URI`.
It will look something like the following. The username and password, if not provided,
can be configured in _Database Access_ under Security in the left panel.

```
export MONGODB_URI="mongodb+srv://<username>:<password>@cluster0.foo.mongodb.net/?retryWrites=true&w=majority"
```

Head to Atlas UI to find the connection string.

NOTE: There are a number of ways to navigate the Atlas UI. Keep your eye out for "Connect" and "driver".

On the left panel, find and click 'Database' under DEPLOYMENT.
Click the Connect button that appears, then Drivers. Select Python.
(Have no concern for the version. This is the PyMongo, not Python, version.)
Once you have the Connect Window open, you will see an instruction to `pip install pymongo`.
You will also see a **connection string**.
This is the `uri` that a `pymongo.MongoClient` uses to connect to the Database.

### Test the connection

Atlas provides a simple check. Once you have your `uri` and `pymongo` installed,
try the following in a python console.

```python
from pymongo.mongo_client import MongoClient

client = MongoClient(uri)  # Create a new client and connect to the server
try:
    client.admin.command(
        "ping"
    )  # Send a ping to confirm a successful connection
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
```

**Troubleshooting**

- You can edit a Database's users and passwords on the 'Database Access' page, under Security.
- Remember to add your IP address. (Try `curl -4 ifconfig.co`)

### Create a Database and Collection

As mentioned, Vector Databases provide two functions. In addition to being the data store,
they provide very efficient search based on natural language queries.
With Vector Search, one will index and query data with a powerful vector search algorithm
using "Hierarchical Navigable Small World (HNSW) graphs to find vector similarity.

The indexing runs beside the data as a separate service asynchronously.
The Search index monitors changes to the Collection that it applies to.
Subsequently, one need not upload the data first.
We will create an empty collection now, which will simplify setup in the example notebook.

Back in the UI, navigate to the Database Deployments page by clicking Database on the left panel.
Click the "Browse Collections" and then "+ Create Database" buttons.
This will open a window where you choose Database and Collection names. (No additional preferences.)
Remember these values as they will be as the environment variables,
`MONGODB_DATABASE` and `MONGODB_COLLECTION`.

### Set Datastore Environment Variables

To establish a connection to the MongoDB Cluster, Database, and Collection, plus create a Vector Search Index,
define the following environment variables.
You can confirm that the required ones have been set like this: `assert "MONGODB_URI" in os.environ`

**IMPORTANT** It is crucial that the choices are consistent between setup in Atlas and Python environment(s).

| Name                 | Description       | Example                                                             |
| -------------------- | ----------------- | ------------------------------------------------------------------- |
| `MONGODB_URI`        | Connection String | mongodb+srv://`<user>`:`<password>`@llama-index.zeatahb.mongodb.net |
| `MONGODB_DATABASE`   | Database name     | llama_index_test_db                                                 |
| `MONGODB_COLLECTION` | Collection name   | llama_index_test_vectorstore                                        |
| `MONGODB_INDEX`      | Search index name | vector_index                                                        |

The following will be required to authenticate with OpenAI.

| Name             | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| `OPENAI_API_KEY` | OpenAI token created at https://platform.openai.com/api-keys |

### Create an Atlas Vector Search Index

The final step to configure MongoDB as the Datastore is to create a Vector Search Index.
The procedure is described [here](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#procedure).

Under Services on the left panel, choose Atlas Search > Create Search Index >
Atlas Vector Search JSON Editor.

The Plugin expects an index definition like the following.
To begin, choose `numDimensions: 1536` along with the suggested EMBEDDING variables above.
You can experiment with these later.

```json
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

### Running MongoDB Integration Tests

In addition to the [Jupyter Notebook](https://docs.llamaindex.ai/en/stable/examples/vector_stores/MongoDBAtlasVectorSearch/) in the documentation,
a suite of integration tests is available to verify the MongoDB integration unders `./tests`.
This test suite needs the cluster up and running, and the environment variables defined above.

### Filter Parameters: `vector_filter` vs `search_filter`

**Performance Note**: Filters are now applied at the database query stage (as part of `$vectorSearch` or `$search` pipeline stages) rather than post-processing results in Python. This significantly reduces data transfer and memory usage, especially for large result sets with restrictive filters.

The integration uses two distinct filtering syntaxes that map to different stages in MongoDB Atlas:

- **Vector pre-filter (MQL)**: Applied in the `$vectorSearch` stage via the `filter` key. Built with standard MongoDB Query Language operators (e.g. `{ "metadata.year": { "$gte": 2020 } }`). Generated by `filters_to_mql`.
- **Full-Text / Atlas Search compound filter**: Applied inside the `$search` stage using Atlas Search clause objects (`equals`, `range`, `in`, `must`, `should`, `mustNot`, `minimumShouldMatch`). Generated by `filters_to_atlas_search_compound` and passed as `search_filter`.

To avoid ambiguity, the full-text helper `fulltext_search_stage` now prefers the parameter name `search_filter`. For **backward compatibility**, legacy callers that still pass `filter=` will be transparently mapped to `search_filter` unless both are provided (in which case a `ValueError` is raised).

Example (Vector search with MQL filter):

```python
vector_filter = {"metadata.year": {"$gte": 2020}}
stage = vector_search_stage(
  query_vector=embedding,
  search_field="embedding",
  index_name="vector_index",
  limit=5,
  filter=vector_filter,  # MQL pre-filter
)
```

Example (Full-text search with compound filter):

```python
search_filter = {
  "must": [
    {"equals": {"path": "metadata.genre", "value": "Comedy"}},
    {"range": {"path": "metadata.year", "gte": 2020}},
  ],
  "should": [
    {"equals": {"path": "metadata.language", "value": "en"}}
  ],
  "minimumShouldMatch": 1,
}
pipeline = fulltext_search_stage(
  query="funny scenes",
  search_field="text",
  index_name="search_index",
  operator="text",
  search_filter=search_filter,
)
```

Legacy usage still works:

```python
pipeline = fulltext_search_stage(
  query="funny scenes",
  search_field="text",
  index_name="search_index",
  operator="text",
  filter=search_filter,  # legacy name, automatically mapped
)
```

If both `filter` and `search_filter` are supplied a `ValueError` is raised to prevent ambiguous intent.

### Test Coverage of Filters

Unit tests (`tests/test_mongodb_pipelines.py`) exercise:

- AND vs OR logic translation for Atlas Search compound filters
- Negative operators (`NE`, `NIN`) routed to `mustNot`
- Range and equality clauses
- Omission of empty pre-filter for `$vectorSearch`
- Backward compatibility shim (`filter` alias) and conflict error path

Run just these tests:

```bash
pytest llama-index-integrations/vector_stores/llama-index-vector-stores-mongodb/tests/test_mongodb_pipelines.py -q
```

### Structural Emptiness: `FilterOperator.IS_EMPTY`

The `IS_EMPTY` operator lets you match documents where a metadata field is structurally "empty". A field is considered empty if:

- It is **missing** entirely from the document
- It is explicitly set to **`null`**
- It exists and is an **empty array `[]`**

This tri-state interpretation is useful when upstream ingestion may omit a key, store `null`, or normalize to an empty list interchangeably.

Implementation details:

| Layer | Translation |
|-------|-------------|
| MQL pre-filter (`filters_to_mql`) | Expands one `IS_EMPTY` into a nested `$or` with three branches: `{key: {"$exists": false}}`, `{key: None}`, `{key: []}` |
| Atlas Search (`filters_to_atlas_search_compound`) | Builds a nested compound: two `equals` clauses (null, empty array) plus a `mustNot exists` clause for the missing case. In AND contexts it's wrapped in a `compound.should` group with `minimumShouldMatch: 1`; in OR contexts each branch is added to the top-level `should`. |

Usage examples:

Match documents where `metadata.country` is empty (missing/null/[]):

```python
filters = MetadataFilters(
  filters=[
    MetadataFilter(
      key="country",  # logical key under metadata
      value=None,      # pydantic requires value field; ignored for IS_EMPTY
      operator=FilterOperator.IS_EMPTY,
    )
  ]
)
vector_filter = filters_to_mql(filters)
# vector_filter example output:
# {"$or": [
#   {"metadata.country": {"$exists": False}},
#   {"metadata.country": None},
#   {"metadata.country": []}
# ]}
```

Combine allowed list OR emptiness (value present in list OR field missing/null/empty array):

```python
filters = MetadataFilters(
  filters=[
    MetadataFilter(
      key="country",
      value=["FR", "CA"],
      operator=FilterOperator.IN,
    ),
    MetadataFilter(
      key="country",
      value=None,  # ignored for IS_EMPTY
      operator=FilterOperator.IS_EMPTY,
    ),
  ],
  condition=FilterCondition.OR,
)
```

AND combination (e.g. empty country AND year >= 2024):

```python
filters = MetadataFilters(
  filters=[
    MetadataFilter(key="country", value=None, operator=FilterOperator.IS_EMPTY),
    MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE),
  ],
  condition=FilterCondition.AND,
)
```

Notes & caveats:

- Equality to an empty array matches only a field stored as `[]`; if your ingestion uses `null` or omission instead, those branches still cover you.
- `IS_EMPTY` is structural and therefore not mapped in `map_lc_mql_filter_operators`; attempting to call that mapping directly with `IS_EMPTY` will raise.
- Provide `value=None` when constructing the `MetadataFilter` to satisfy pydantic, though the value is ignored.
- If you need to exclude just one emptiness form (e.g. treat missing as different from `null`), use explicit negative filters (`NE`, `NIN`) or add a custom ingestion normalization step before indexing.

