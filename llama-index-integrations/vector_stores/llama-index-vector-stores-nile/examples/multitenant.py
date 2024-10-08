# Example usage of the NileVectorStore
# We'll use some example data to test the store, you can get it here:
# wget --user-agent "Mozilla" "https://raw.githubusercontent.com/niledatabase/niledatabase/main/examples/ai/sales_insight/data/transcripts/nexiv-solutions__0_transcript.txt" -O "nexiv-solutions__0_transcript.txt"
# wget --user-agent "Mozilla" "https://raw.githubusercontent.com/niledatabase/niledatabase/main/examples/ai/sales_insight/data/transcripts/modamart__0_transcript.txt" -O "modamart__0_transcript.txt"
# These are two call transcripts from sales calls that belong to two different companies

from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.nile import NileVectorStore

# Create a NileVectorStore instance
vector_store = NileVectorStore(
    service_url = "postgresql://01926a52-3859-7dea-b219-b300b7bb0328:3a98e4eb-f519-483c-bd8b-6a79cd5674a8@us-west-2.db.thenile.dev:5432/niledb",
    table_name = "test_table",
    tenant_aware = True,
    num_dimensions = 1536
)

# Load the data
reader = SimpleDirectoryReader(input_files=["nexiv-solutions__0_transcript.txt"])
documents_nexiv = reader.load_data()

reader = SimpleDirectoryReader(input_files=["modamart__0_transcript.txt"])
documents_modamart = reader.load_data()

tenant_id_nexiv = vector_store.create_tenant("nexiv-solutions")
tenant_id_modamart = vector_store.create_tenant("modamart")

# Add the tenant id to the metadata
for doc in documents_nexiv:
    doc.metadata["tenant_id"] = tenant_id_nexiv

for doc in documents_modamart:
    doc.metadata["tenant_id"] = tenant_id_modamart

# store data and embeddings
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents_nexiv + documents_modamart, storage_context=storage_context, show_progress=True
)

# Query the data
nexiv_query_engine = index.as_query_engine(similarity_top_k=3, tenant_id=tenant_id_nexiv)
modamart_query_engine = index.as_query_engine(similarity_top_k=3, tenant_id=tenant_id_modamart)
print("test query on nexiv: ", nexiv_query_engine.query("What were the customer pain points?"))
print("test query on modamart: ", modamart_query_engine.query("What were the customer pain points?"))

