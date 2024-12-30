# Example usage of the NileVectorStore
# We'll use some example data to test the store, you can get it here:
# wget --user-agent "Mozilla" "https://raw.githubusercontent.com/niledatabase/niledatabase/main/examples/ai/sales_insight/data/transcripts/nexiv-solutions__0_transcript.txt" -O "nexiv-solutions__0_transcript.txt"
# wget --user-agent "Mozilla" "https://raw.githubusercontent.com/niledatabase/niledatabase/main/examples/ai/sales_insight/data/transcripts/modamart__0_transcript.txt" -O "modamart__0_transcript.txt"
# These are two call transcripts from sales calls that belong to two different companies
# You will also need to set OPENAI_API_KEY environment variable prior to running this example

import logging

logging.basicConfig(level=logging.INFO)

from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.vector_stores.nile import NileVectorStore, IndexType

# Create a NileVectorStore instance
vector_store = NileVectorStore(
    service_url="postgresql://user:password@us-west-2.db.thenile.dev:5432/niledb",
    table_name="test_table",
    tenant_aware=True,
    num_dimensions=1536,
)

# Load the data
reader = SimpleDirectoryReader(input_files=["nexiv-solutions__0_transcript.txt"])
documents_nexiv = reader.load_data()

reader = SimpleDirectoryReader(input_files=["modamart__0_transcript.txt"])
documents_modamart = reader.load_data()

tenant_id_nexiv = str(vector_store.create_tenant("nexiv-solutions"))
tenant_id_modamart = str(vector_store.create_tenant("modamart"))

# Add the tenant id to the metadata
for i, doc in enumerate(documents_nexiv, start=1):
    doc.metadata["tenant_id"] = tenant_id_nexiv
    doc.metadata["category"] = "IT"  # This is just to test the filter
    doc.id_ = f"nexiv_doc_id_{i}"  # This is for testing the delete function

for i, doc in enumerate(documents_modamart, start=1):
    doc.metadata["tenant_id"] = tenant_id_modamart
    doc.metadata["category"] = "Retail"
    doc.id_ = f"modamart_doc_id_{i}"

# store data and embeddings
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents_nexiv + documents_modamart,
    storage_context=storage_context,
    show_progress=True,
)

# Create a vector index (optional, the default index is flat also known as no index)
# Note that this makes no sense to do for this tiny example, but it's a good idea to create an index for large datasets
try:
    vector_store.create_index(index_type=IndexType.PGVECTOR_IVFFLAT, nlists=10)
except Exception as e:
    # This will throw an error if the index already exists, which is expected in this case
    print(e)

# Query the data
nexiv_query_engine = index.as_query_engine(
    similarity_top_k=3,
    vector_store_kwargs={
        "tenant_id": str(tenant_id_nexiv),
        "ivfflat_probes": 10,  # optional, this is only needed for the PGVECTOR_IVFFLAT index
    },
)
modamart_query_engine = index.as_query_engine(
    similarity_top_k=3,
    vector_store_kwargs={
        "tenant_id": str(tenant_id_modamart),
    },
)
print(
    "test query on nexiv: ",
    nexiv_query_engine.query("What were the customer pain points?"),
)
print(
    "test query on modamart: ",
    modamart_query_engine.query("What were the customer pain points?"),
)

# Query with a filter
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="category", operator=FilterOperator.EQ, value="Retail"),
    ]
)

nexiv_query_engine_filtered = index.as_query_engine(
    similarity_top_k=3,
    filters=filters,
    vector_store_kwargs={"tenant_id": str(tenant_id_nexiv)},
)
print(
    "test query on nexiv with filter on category = Retail (should return empty): ",
    nexiv_query_engine_filtered.query("What were the customer pain points?"),
)

# Delete a document from the store
ref_doc_id = "nexiv_doc_id_1"

# Use this key to delete a document from the store, using the correct tenant_id
print("deleting document: ", ref_doc_id, " with tenant_id: ", tenant_id_nexiv)
vector_store.delete(ref_doc_id, tenant_id=tenant_id_nexiv)

# Query the data again
print(
    "test query on nexiv after deletion (should return empty): ",
    nexiv_query_engine.query("What were the customer pain points?"),
)
