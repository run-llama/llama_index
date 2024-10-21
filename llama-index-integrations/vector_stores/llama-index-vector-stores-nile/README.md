# Nile Vector Store (PostgreSQL)

This integration makes it possible to use [Nile - Postgres re-engineered for multi-tenant applications](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/)
as a vector store in LlamaIndex.

## What is Nile?

Nile is a Postgres database that enables all database operations per tenant including auto-scaling, branching, and backups, with full customer isolation.

Multi-tenant RAG applications are increasingly popular, since they provide security and privacy while using large language models.

However, managing the underlying Postgres database is not straightforward. DB-per-tenant is expensive and complex to manage, while shared-DB has security and privacy concerns, and also limits the scalability and performance of the RAG application. Nile re-engineered Postgres to deliver the best of all worlds - the isolation of DB-per-tenant, at the cost, efficiency and developer experience of a shared-DB.

Storing millions of vectors in a shared-DB can be slow and require significant resources to index and query. But if you store 1000 tenants in Nile's virtual tenant databases, each with 1000 vectors, this can be quite manageable. Especially since you can place larger tenants on their own compute, while smaller tenants can efficiently share compute resources and auto-scale as needed.

## Getting Started with Nile

Start by signing up for [Nile](https://console.thenile.dev/?utm_campaign=partnerlaunch&utm_source=llamaindex&utm_medium=docs). Once you've signed up for Nile, you'll be promoted to create your first database. Go ahead and do so. You'll be redirected to the "Query Editor" page of your new database.

From there, click on "Home" (top icon on the left menu), click on "generate credentials" and copy the resulting connection string. You will need it in a sec.

## Quickstart

Install the integration with:

```bash
pip install llama-index-vector-stores-nile
```

Use the connection string you generated earlier (at the "Getting started" step) to create a tenant-aware vector store.

:fire: NileVectorStore supports both tenant-aware vector stores, that isolates the documents for each tenant and a regular store which is typically used for shared data that all tenants can access. Below, we'll demonstrate the tenant-aware vector store.

```python
# Replace with your connection string.
NILE_SERVICE_URL = "postgresql://nile:password@db.thenile.dev:5432/nile"

vector_store = NileVectorStore(
    service_url=NILEDB_SERVICE_URL,
    table_name="documents",
    tenant_aware=True,
    num_dimensions=1536,
)
```

Create an index from documents:

```python
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents_nexiv + documents_modamart,
    storage_context=storage_context,
    show_progress=True,
)
```

Or from existing embeddings:

```python
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
```

and query each tenant's data with guaranteed isolation:

```python
query_engine = index.as_query_engine(
    vector_store_kwargs={
        "tenant_id": str(tenant_id_modamart),
    },
)
response = query_engine.query("What action items do we need to follow up on?")

print(response)
```

See resources below for more information and examples.

## Additional Resources

- [Example iPython / Jupyter notebook for Nile and LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/vector_stores/NileVectorStore/)
- [Nile's generative AI and vector embeddings docs](https://www.thenile.dev/docs/ai-embeddings)
- [Nile's LlamaIndex documentation](https://www.thenile.dev/docs/partners/llama)
- [Nile's pgvector primer](https://www.thenile.dev/docs/ai-embeddings/pg_vector)
- [Few things you didn't know about pgvector](https://www.thenile.dev/blog/pgvector_myth_debunking)
