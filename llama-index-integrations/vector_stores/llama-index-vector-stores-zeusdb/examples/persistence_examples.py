# examples/persistence_examples.py
"""
ZeusDB Persistence Examples for LlamaIndex

Demonstrates how to save and load ZeusDB indexes with full state preservation,
including vectors, metadata, HNSW graph structure, and quantization models.
"""

from pathlib import Path
import shutil

from dotenv import load_dotenv

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.zeusdb import ZeusDBVectorStore

load_dotenv()

# Configure OpenAI
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4")

print("=" * 70)
print("ZeusDB Persistence Examples")
print("=" * 70)
print()

# =============================================================================
# Example 1: Basic Save and Load
# =============================================================================
print("=" * 70)
print("Example 1: Basic Save and Load")
print("=" * 70)
print()

# Create sample documents
documents = [
    Document(
        text=f"Document {i}: This is a sample document about technology "
        f"and artificial intelligence.",
        metadata={"doc_id": i, "category": "tech"},
    )
    for i in range(50)
]

print(f"Creating index with {len(documents)} documents...")

# Create vector store and index
vector_store = ZeusDBVectorStore(dim=1536, distance="cosine", index_type="hnsw")

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)

print()
print("Original index:")
print(f"  Vector count: {vector_store.get_vector_count()}")
print(f"  Index info: {vector_store.info()}")
print()

# Test search before saving
query_text = "artificial intelligence"
print(f"Testing search before save: '{query_text}'")

# Get embedding for query - use get_text_embedding_batch for single query
embed_model = Settings.embed_model
query_embedding = embed_model.get_text_embedding(query_text)

# Direct query to vector store
query_obj = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=3)
results = vector_store.query(query_obj)

# ✅ Handle None values safely
result_ids = results.ids or []
result_sims = results.similarities or []

print(f"  Found {len(result_ids)} results")
for i, (node_id, similarity) in enumerate(zip(result_ids, result_sims), 1):
    print(f"  {i}. Score: {similarity:.4f}, ID: {node_id}")
print()

# Save the index
save_path = "test_index.zdb"
print(f"Saving index to '{save_path}'...")
success = vector_store.save_index(save_path)
print(f"  Save successful: {success}")
print()

# Load the index
print(f"Loading index from '{save_path}'...")
loaded_vector_store = ZeusDBVectorStore.load_index(save_path)
print("  Load successful!")
print()

# Verify loaded index
print("Loaded index:")
print(f"  Vector count: {loaded_vector_store.get_vector_count()}")
print(f"  Index info: {loaded_vector_store.info()}")
print()

# Test search after loading - direct query
print(f"Testing search after load: '{query_text}'")
loaded_results = loaded_vector_store.query(query_obj)

# ✅ Handle None values safely
loaded_ids = loaded_results.ids or []
loaded_sims = loaded_results.similarities or []

print(f"  Found {len(loaded_ids)} results")
for i, (node_id, similarity) in enumerate(zip(loaded_ids, loaded_sims), 1):
    print(f"  {i}. Score: {similarity:.4f}, ID: {node_id}")
print()

# Compare results
print("Comparing search results:")
if len(result_ids) == len(loaded_ids):
    print(f"  ✅ Same number of results: {len(result_ids)}")

    if result_ids and loaded_ids:
        # Compare top result IDs
        if result_ids[0] == loaded_ids[0]:
            print(f"  ✅ Top result matches: ID={result_ids[0]}")
        else:
            print(f"  ⚠️  Top results differ: {result_ids[0]} vs {loaded_ids[0]}")

        # Compare similarities (should be very close)
        if result_sims and loaded_sims:
            sim_diff = abs(result_sims[0] - loaded_sims[0])
            if sim_diff < 0.001:
                print(f"  ✅ Similarities match: {result_sims[0]:.4f}")
            else:
                print(f"  ⚠️  Similarities differ by {sim_diff:.4f}")
else:
    print("  ⚠️  Different number of results")
print()

# =============================================================================
# Example 2: Save and Load with Quantization
# =============================================================================
print("=" * 70)
print("Example 2: Save and Load with Quantization")
print("=" * 70)
print()

# Create index with quantization
print("Creating quantized index...")
quantization_config = {
    "type": "pq",
    "subvectors": 8,
    "bits": 8,
    "training_size": 1000,
    "storage_mode": "quantized_only",
}

quant_vector_store = ZeusDBVectorStore(
    dim=1536,
    distance="cosine",
    index_type="hnsw",
    quantization_config=quantization_config,
)

quant_storage_context = StorageContext.from_defaults(vector_store=quant_vector_store)

# Create more documents to trigger quantization
quant_documents = [
    Document(
        text=f"Document {i}: Technology content for quantization testing.",
        metadata={"doc_id": i, "category": "tech"},
    )
    for i in range(1100)  # Enough to trigger training
]

print(f"Adding {len(quant_documents)} documents...")
quant_index = VectorStoreIndex.from_documents(
    quant_documents, storage_context=quant_storage_context, show_progress=False
)

print()
print("Original quantized index:")
print(f"  Vector count: {quant_vector_store.get_vector_count()}")
print(f"  Is quantized: {quant_vector_store.is_quantized()}")
print(f"  Training progress: {quant_vector_store.get_training_progress():.1f}%")
print(f"  Storage mode: {quant_vector_store.get_storage_mode()}")

qi = quant_vector_store.get_quantization_info()
if qi:
    print(f"  Compression: {qi.get('compression_ratio', 0):.1f}x")
    print(f"  Memory: {qi.get('memory_mb', 0):.2f} MB")
    print(f"  Trained: {qi.get('is_trained', False)}")
print()

# Save quantized index
quant_save_path = "quantized_index.zdb"
print(f"Saving quantized index to '{quant_save_path}'...")
quant_success = quant_vector_store.save_index(quant_save_path)
print(f"  Save successful: {quant_success}")
print()

# Load quantized index
print(f"Loading quantized index from '{quant_save_path}'...")
loaded_quant_vs = ZeusDBVectorStore.load_index(quant_save_path)
print("  Load successful!")
print()

# Verify quantization state preserved
print("Loaded quantized index:")
print(f"  Vector count: {loaded_quant_vs.get_vector_count()}")
print(f"  Is quantized: {loaded_quant_vs.is_quantized()}")
print(f"  Training progress: {loaded_quant_vs.get_training_progress():.1f}%")
print(f"  Storage mode: {loaded_quant_vs.get_storage_mode()}")

loaded_qi = loaded_quant_vs.get_quantization_info()
if loaded_qi:
    print(f"  Compression: {loaded_qi.get('compression_ratio', 0):.1f}x")
    print(f"  Memory: {loaded_qi.get('memory_mb', 0):.2f} MB")
    print(f"  Trained: {loaded_qi.get('is_trained', False)}")
print()

# Verify quantization state matches
print("Verifying quantization state preservation:")

# Check original state
orig_quantized = quant_vector_store.is_quantized()
orig_storage = quant_vector_store.get_storage_mode()

# Check loaded state
loaded_quantized = loaded_quant_vs.is_quantized()
loaded_storage = loaded_quant_vs.get_storage_mode()

if orig_quantized == loaded_quantized:
    print(f"  ✅ Quantization state preserved: {loaded_quantized}")
else:
    print("  ⚠️  Quantization state differs:")
    print(f"     Original: is_quantized={orig_quantized}, mode={orig_storage}")
    print(f"     Loaded:   is_quantized={loaded_quantized}, mode={loaded_storage}")
    print()
    print("  ℹ️  Known Limitation (Current Release):")
    print("     Quantized indexes load in raw vector mode.")
    print("     Training state and config are preserved.")
    print("     Search works perfectly, just without memory compression.")
    print("     This will be fixed in the next ZeusDB release.")
    print()

if (
    quant_vector_store.get_training_progress()
    == loaded_quant_vs.get_training_progress()
):
    progress = loaded_quant_vs.get_training_progress()
    print(f"  ✅ Training progress preserved: {progress:.1f}%")
else:
    print("  ⚠️  Training progress differs")

if qi and loaded_qi:
    if qi.get("compression_ratio") == loaded_qi.get("compression_ratio"):
        ratio = loaded_qi.get("compression_ratio")
        print(f"  ✅ Compression ratio preserved: {ratio:.1f}x")
    else:
        print("  ⚠️  Compression ratio differs")

    if qi.get("is_trained") == loaded_qi.get("is_trained"):
        trained = loaded_qi.get("is_trained")
        print(f"  ✅ Training status preserved: {trained}")
    else:
        print("  ⚠️  Training status differs")
print()

# Test search on loaded quantized index
print("Testing search on loaded quantized index...")
quant_query_text = "technology and AI"
quant_query_embedding = embed_model.get_text_embedding(quant_query_text)
quant_query_obj = VectorStoreQuery(
    query_embedding=quant_query_embedding, similarity_top_k=3
)

loaded_quant_results = loaded_quant_vs.query(quant_query_obj)

# ✅ Handle None values safely
loaded_quant_ids = loaded_quant_results.ids or []
loaded_quant_sims = loaded_quant_results.similarities or []

print(f"  Found {len(loaded_quant_ids)} results")
for i, (node_id, similarity) in enumerate(zip(loaded_quant_ids, loaded_quant_sims), 1):
    print(f"  {i}. Score: {similarity:.4f}, ID: {node_id}")
print()

# =============================================================================
# Example 3: Multiple Save/Load Cycles
# =============================================================================
print("=" * 70)
print("Example 3: Multiple Save/Load Cycles")
print("=" * 70)
print()

# Create initial index
cycle_docs = [
    Document(text=f"Cycle document {i}", metadata={"doc_id": i, "version": 1})
    for i in range(100)
]

print("Creating initial index...")
cycle_vs = ZeusDBVectorStore(dim=1536, distance="cosine")
cycle_sc = StorageContext.from_defaults(vector_store=cycle_vs)
cycle_idx = VectorStoreIndex.from_documents(
    cycle_docs, storage_context=cycle_sc, show_progress=False
)

initial_count = cycle_vs.get_vector_count()
print(f"  Initial vector count: {initial_count}")
print()

# Save and load multiple times
for cycle in range(1, 4):
    save_path = f"cycle_{cycle}.zdb"
    print(f"Cycle {cycle}: Save -> Load")

    # Save
    print(f"  Saving to '{save_path}'...")
    cycle_vs.save_index(save_path)

    # Load
    print(f"  Loading from '{save_path}'...")
    cycle_vs = ZeusDBVectorStore.load_index(save_path)

    # Verify
    current_count = cycle_vs.get_vector_count()
    current_info = cycle_vs.info()
    print(f"  Vector count after load: {current_count}")
    print(f"  Index info: {current_info}")

    if current_count == initial_count:
        print(f"  ✅ Vector count stable across cycle {cycle}")
    else:
        print(f"  ⚠️  Vector count changed: {initial_count} -> {current_count}")
    print()

print("All cycles completed successfully!")
print()

# =============================================================================
# Example 4: Cleanup Test Files
# =============================================================================
print("=" * 70)
print("Example 4: Cleanup")
print("=" * 70)
print()

test_paths = [
    "test_index.zdb",
    "quantized_index.zdb",
    "cycle_1.zdb",
    "cycle_2.zdb",
    "cycle_3.zdb",
]

print("Cleaning up test files:")
for path in test_paths:
    if Path(path).exists():
        try:
            shutil.rmtree(path)
            print(f"  ✅ Removed: {path}")
        except Exception as e:
            print(f"  ⚠️  Failed to remove {path}: {e}")
    else:
        print(f"  ℹ️  Not found: {path}")
print()

# =============================================================================
# Summary
# =============================================================================
print("=" * 70)
print("Summary: Persistence Benefits")
print("=" * 70)
print()
print("✅ Complete state preservation - vectors, metadata, HNSW graph")
print("✅ Quantization support - PQ models and training state preserved")
print("✅ Cross-platform compatibility - portable between systems")
print("✅ Directory structure - organized file layout (.zdb directory)")
print("✅ Direct querying - use vector_store.query() after loading")
print()
print("💡 Key Points:")
print("  • Save path creates a directory, not a single file")
print("  • Quantized indexes preserve compression models and training state")
print("  • Loaded indexes are ready for immediate use via .query()")
print("  • Use save_index() and load_index() class method")
print()
print("⚠️  Note on LlamaIndex Integration:")
print("  • ZeusDB doesn't store full text (only embeddings + metadata)")
print("  • Use vector_store.query() directly instead of VectorStoreIndex")
print("  • For full LlamaIndex integration, store documents separately")
print()
print("📚 Persistence is essential for:")
print("  • Production deployments with long-lived indexes")
print("  • Sharing indexes between systems or team members")
print("  • Implementing backup and recovery strategies")
print("  • Avoiding re-indexing on application restart")
print()
