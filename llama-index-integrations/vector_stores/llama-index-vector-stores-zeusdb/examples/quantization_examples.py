# examples/quantization_examples.py
"""
ZeusDB Product Quantization Examples for LlamaIndex

Demonstrates how to use Product Quantization (PQ) for memory-efficient
vector storage with the LlamaIndex ZeusDB integration.
"""

from dotenv import load_dotenv

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.zeusdb import ZeusDBVectorStore

load_dotenv()

# Configure OpenAI
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4")  # ✅ Fixed: Changed from gpt-5 to gpt-4

print("=" * 70)
print("ZeusDB Product Quantization Examples")
print("=" * 70)
print()

# =============================================================================
# Example 1: Basic Quantization Setup
# =============================================================================
print("=" * 70)
print("Example 1: Basic Quantization Configuration")
print("=" * 70)
print()

# Configure quantization for memory efficiency
quantization_config = {
    "type": "pq",  # Product Quantization
    "subvectors": 8,  # Divide 1536-dim into 8 subvectors
    "bits": 8,  # 256 centroids per subvector (2^8)
    "training_size": 1000,  # Minimum: 1000 (required by backend)
    "storage_mode": "quantized_only",  # Memory-optimized mode
}

print("Quantization Config:")
for key, value in quantization_config.items():
    print(f"  {key}: {value}")
print()

# Create vector store with quantization
vector_store = ZeusDBVectorStore(
    dim=1536,
    distance="cosine",
    index_type="hnsw",
    quantization_config=quantization_config,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Initial State:")
print(f"  Vector count: {vector_store.get_vector_count()}")
print(f"  Is quantized: {vector_store.is_quantized()}")
print(f"  Can use quantization: {vector_store.can_use_quantization()}")
print(f"  Storage mode: {vector_store.get_storage_mode()}")
print(f"  Training progress: {vector_store.get_training_progress():.1f}%")
print()

# =============================================================================
# Example 2: Adding Documents and Triggering Training
# =============================================================================
print("=" * 70)
print("Example 2: Adding Documents (No Training in This Example)")
print("=" * 70)
print()

# Create sample documents
documents = [
    Document(
        text=f"Document {i}: This is a sample document about technology, "
        f"artificial intelligence, and machine learning in the year {2020 + (i % 5)}.",
        metadata={"doc_id": i, "category": "tech", "year": 2020 + (i % 5)},
    )
    for i in range(150)  # Small sample - won't trigger training (need 1000)
]

print(f"Adding {len(documents)} documents...")
print("⚠️  Note: Training requires 1000 documents, so it won't trigger here")
print("   (See Example 4 for actual training demonstration)")
print()

# Build index - this will add all documents
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)

print()
print("After Adding Documents:")
print(f"  Vector count: {vector_store.get_vector_count()}")
print(f"  Is quantized: {vector_store.is_quantized()}")
print(f"  Can use quantization: {vector_store.can_use_quantization()}")
print(f"  Storage mode: {vector_store.get_storage_mode()}")
print(f"  Training progress: {vector_store.get_training_progress():.1f}%")
print()

# Get detailed quantization info
quant_info = vector_store.get_quantization_info()
if quant_info:
    print("Quantization Details:")
    for key, value in quant_info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print()
else:
    print("ℹ️  Quantization info not available (training not yet triggered)")
    print()

# =============================================================================
# Example 3: Querying with Index
# =============================================================================
print("=" * 70)
print("Example 3: Searching (Without Quantization)")
print("=" * 70)
print()

query_engine = index.as_query_engine(similarity_top_k=3)

query = "Tell me about artificial intelligence and machine learning"
print(f"Query: {query}")
print()

response = query_engine.query(query)
print(f"Response: {response}")
print()

# Also try direct retrieval
retriever = index.as_retriever(similarity_top_k=5)
results = retriever.retrieve("technology and AI")

print("Direct Retrieval Results (top 5):")
for i, node in enumerate(results, 1):
    print(f"{i}. Score: {node.score:.4f}")
    print(f"   Text: {node.text[:80]}...")
    print(f"   Metadata: {node.metadata}")
    print()

# =============================================================================
# Example 4: Comparing Different Quantization Configurations
# =============================================================================
print("=" * 70)
print("Example 4: Comparing Quantization Configurations")
print("=" * 70)
print()

configs = [
    {
        "name": "High Compression (Memory Optimized)",
        "config": {
            "type": "pq",
            "subvectors": 16,
            "bits": 6,
            "training_size": 1000,
            "storage_mode": "quantized_only",
        },
        "description": "~32x compression, lowest memory usage",
    },
    {
        "name": "Balanced (Recommended)",
        "config": {
            "type": "pq",
            "subvectors": 8,
            "bits": 8,
            "training_size": 1000,
            "storage_mode": "quantized_only",
        },
        "description": "~16x compression, good accuracy/memory balance",
    },
    {
        "name": "High Accuracy (Keep Raw Vectors)",
        "config": {
            "type": "pq",
            "subvectors": 4,
            "bits": 8,
            "training_size": 1000,
            "storage_mode": "quantized_with_raw",
        },
        "description": "~4x compression, keeps raw vectors",
    },
]

# Create a test dataset - needs to be larger to trigger training
test_docs = [
    Document(
        text=f"Test document {i} about various topics including science, "
        f"technology, and research.",
        metadata={"id": i, "category": "test"},
    )
    for i in range(1100)  # Just over training threshold
]

print("Testing different configurations with same dataset:")
print(f"(Using {len(test_docs)} documents for each configuration)")
print("⚠️  Note: This will take several minutes as it creates 3 indexes")
print("   and trains quantization for each one.")
print()

for config_info in configs:
    print(f"Configuration: {config_info['name']}")
    print(f"  Description: {config_info['description']}")
    print(f"  Settings: {config_info['config']}")

    # Create new vector store with this config
    vs = ZeusDBVectorStore(
        dim=1536,
        distance="cosine",
        index_type="hnsw",
        quantization_config=config_info["config"],
    )

    sc = StorageContext.from_defaults(vector_store=vs)

    print("  Building index (this may take 1-2 minutes)...")
    idx = VectorStoreIndex.from_documents(
        test_docs, storage_context=sc, show_progress=False
    )

    print("  Results:")
    print(f"    Vector count: {vs.get_vector_count()}")
    print(f"    Is quantized: {vs.is_quantized()}")
    print(f"    Can use quantization: {vs.can_use_quantization()}")
    print(f"    Storage mode: {vs.get_storage_mode()}")
    print(f"    Training progress: {vs.get_training_progress():.1f}%")

    qi = vs.get_quantization_info()
    if qi:
        if "compression_ratio" in qi:
            print(f"    Compression ratio: {qi['compression_ratio']:.1f}x")
        if "memory_mb" in qi:
            print(f"    Memory usage: {qi['memory_mb']:.2f} MB")
        if "is_trained" in qi:
            print(f"    Training complete: {qi['is_trained']}")

    print()

# =============================================================================
# Example 5: Monitoring Training Progress
# =============================================================================
print("=" * 70)
print("Example 5: Monitoring Training Progress")
print("=" * 70)
print()

# Create fresh vector store
monitor_vs = ZeusDBVectorStore(
    dim=1536,
    distance="cosine",
    index_type="hnsw",
    quantization_config={
        "type": "pq",
        "subvectors": 8,
        "bits": 8,
        "training_size": 1000,
        "storage_mode": "quantized_only",
    },
)

monitor_sc = StorageContext.from_defaults(vector_store=monitor_vs)

print("Creating and adding documents to monitor training progress...")
print("Note: Training triggers at 1000 documents")
print()

# Create all documents
print("Creating 1200 documents...")
all_docs = [
    Document(
        text=f"Document {i}: Technology and AI content for training demonstration.",
        metadata={"id": i, "category": "demo"},
    )
    for i in range(1200)
]

print("Before adding:")
print(f"  Vector count: {monitor_vs.get_vector_count()}")
print(f"  Training progress: {monitor_vs.get_training_progress():.1f}%")
print(f"  Is quantized: {monitor_vs.is_quantized()}")
print(f"  Storage mode: {monitor_vs.get_storage_mode()}")
print()

# ✅ Split documents into first batch and remaining
batch_size = 300
first_batch = all_docs[:batch_size]
remaining_docs = all_docs[batch_size:]

print(f"Adding {len(all_docs)} documents in batches of {batch_size}...")
print()

# ✅ Create index with first batch (now monitor_idx is always defined)
print(f"Batch 1: Creating index with {len(first_batch)} documents...", end=" ")
monitor_idx = VectorStoreIndex.from_documents(
    first_batch, storage_context=monitor_sc, show_progress=False
)

count = monitor_vs.get_vector_count()
progress = monitor_vs.get_training_progress()
is_quantized = monitor_vs.is_quantized()

print("Done!")
print(f"  Vectors: {count}, Progress: {progress:.1f}%, Quantized: {is_quantized}")
print()

# ✅ Add remaining documents in batches
remaining_batches = [
    remaining_docs[i : i + batch_size]
    for i in range(0, len(remaining_docs), batch_size)
]

for batch_num, batch in enumerate(remaining_batches, start=2):
    print(f"Batch {batch_num}: Adding {len(batch)} documents...", end=" ")
    for doc in batch:
        monitor_idx.insert(doc)

    count = monitor_vs.get_vector_count()
    progress = monitor_vs.get_training_progress()
    is_quantized = monitor_vs.is_quantized()

    print("Done!")
    print(f"  Vectors: {count}, Progress: {progress:.1f}%, Quantized: {is_quantized}")

    if is_quantized and progress == 100.0:
        print("  🎉 Training completed!")
        qi = monitor_vs.get_quantization_info()
        if qi and "compression_ratio" in qi:
            print(f"  Compression: {qi['compression_ratio']:.1f}x")
    print()

print("Final state:")
print(f"  Vector count: {monitor_vs.get_vector_count()}")
print(f"  Training progress: {monitor_vs.get_training_progress():.1f}%")
print(f"  Is quantized: {monitor_vs.is_quantized()}")
print(f"  Storage mode: {monitor_vs.get_storage_mode()}")
print()

# =============================================================================
# Summary
# =============================================================================
print("=" * 70)
print("Summary: Product Quantization Benefits")
print("=" * 70)
print()
print("✅ Memory Reduction: 4x-256x depending on configuration")
print("✅ Automatic Training: Triggers at configured threshold (minimum 1000)")
print("✅ Transparent Search: Works seamlessly with quantized vectors")
print("✅ Monitoring: Track training progress and quantization status")
print()
print("💡 Recommendations:")
print("  • Start with balanced config (subvectors=8, bits=8)")
print("  • Use 'quantized_only' for maximum memory savings")
print("  • Set training_size to 1000-10000 based on dataset size")
print("  • Monitor training progress with get_training_progress()")
print()
print("⚠️  Important:")
print("  • Minimum training_size is 1000 (enforced by backend)")
print("  • Training occurs once when threshold is reached")
print("  • Larger training sets generally produce better quantization")
print()
print("📚 For large datasets (100k+ vectors), quantization provides")
print("   significant memory savings with minimal accuracy impact.")
print()
