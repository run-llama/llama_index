# Azure Cosmos DB for NoSQL Vector Store

This integration makes possible to use [Azure Cosmos DB for NoSQL](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/)
as a vector store in LlamaIndex.

## Quick start

Install the integration with:

```sh
pip install llama-index-vector-stores-azurecosmosnosql
```

Create the CosmosDB client:

```python
URI = "AZURE_COSMOSDB_URI"
KEY = "AZURE_COSMOSDB_KEY"
client = CosmosClient(URI, credential=KEY)
```

Specify the vector store properties:

```python
indexing_policy = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
}

vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 3072,
        }
    ]
}
```

Create the vector store:

```python
store = AzureCosmosDBNoSqlVectorSearch(
    cosmos_client=client,
    vector_embedding_policy=vector_embedding_policy,
    indexing_policy=indexing_policy,
    cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
    cosmos_database_properties={},
    create_container=True,
)
```

Finally, create the index from a list containing documents:

```python
storage_context = StorageContext.from_defaults(vector_store=store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
```
