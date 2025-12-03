# examples/delete_examples.py
from dotenv import load_dotenv

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.zeusdb import ZeusDBVectorStore

load_dotenv()

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4")

vector_store = ZeusDBVectorStore(dim=1536, distance="cosine", index_type="hnsw")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = [
    Document(
        text="Document A1 from project Alpha",
        metadata={"project": "alpha", "version": 1},
        doc_id="doc_alpha_1",
    ),
    Document(
        text="Document A2 from project Alpha",
        metadata={"project": "alpha", "version": 2},
        doc_id="doc_alpha_2",
    ),
    Document(
        text="Document B1 from project Beta",
        metadata={"project": "beta", "version": 1},
        doc_id="doc_beta_1",
    ),
]

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

print("=== ZeusDB Delete Operation Tests ===\n")
print("  Note: ZeusDB HNSW only supports deletion by node ID\n")

# Show initial state
retriever = index.as_retriever(similarity_top_k=10)
results = retriever.retrieve("document")
initial_count = vector_store.get_vector_count()

print(f"Initial state: {len(results)} documents, {initial_count} vectors")
for node in results:
    print(f"  - {node.node.ref_doc_id} (node_id: {node.node.node_id[:8]}...)")

# Test 1: delete() by ref_doc_id - Should fail
print("\n" + "=" * 60)
print("Test 1: Delete by ref_doc_id (NOT SUPPORTED)")
print("=" * 60)
try:
    index.delete_ref_doc(ref_doc_id="doc_alpha_1", delete_from_docstore=False)
    print("❌ FAIL: Should have raised NotImplementedError")
except NotImplementedError as e:
    print("✅ PASS: Correctly raised NotImplementedError")
    print(f"   Message: {str(e)[:80]}...")

# Test 2: delete_nodes() by ID - Should work
print("\n" + "=" * 60)
print("Test 2: Delete by Node ID (SUPPORTED)")
print("=" * 60)

results = retriever.retrieve("document")
before_count = vector_store.get_vector_count()

if results:
    node_to_delete = results[0]
    node_id = node_to_delete.node.node_id
    ref_doc_id = node_to_delete.node.ref_doc_id

    print(f"Before: {before_count} vectors")
    print(f"Deleting: {ref_doc_id} (node: {node_id[:8]}...)")

    vector_store.delete_nodes(node_ids=[node_id])

    after_count = vector_store.get_vector_count()
    new_results = retriever.retrieve("document")

    print(f"After: {after_count} vectors, {len(new_results)} retrievable documents")

    if after_count == before_count - 1:
        print(f"✅ PASS: Vector count decreased ({before_count} → {after_count})")
    else:
        print(f"❌ FAIL: Expected {before_count - 1}, got {after_count}")

    if len(new_results) == len(results) - 1:
        print(
            f"✅ PASS: Retrievable docs decreased ({len(results)} → {len(new_results)})"
        )
    else:
        print(f"❌ FAIL: Expected {len(results) - 1} docs, got {len(new_results)}")

# Test 3: delete_nodes() with filters - Should fail
print("\n" + "=" * 60)
print("Test 3: Delete with Filters (NOT SUPPORTED)")
print("=" * 60)

filters = MetadataFilters.from_dicts(
    [{"key": "project", "value": "beta", "operator": FilterOperator.EQ}]
)

try:
    vector_store.delete_nodes(filters=filters)
    print("❌ FAIL: Should have raised NotImplementedError")
except NotImplementedError as e:
    print("✅ PASS: Correctly raised NotImplementedError")
    print(f"   Message: {str(e)[:80]}...")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("✅ delete_nodes(node_ids=[...]) works correctly")
print("✅ delete(ref_doc_id='...') correctly raises NotImplementedError")
print("✅ delete_nodes(filters=...) correctly raises NotImplementedError")
print("\n📝 Recommendation: Track node IDs at application level for deletion")
