# examples/async_examples.py
"""
ZeusDB Async Examples for LlamaIndex

Demonstrates asynchronous operations for non-blocking, concurrent vector operations.

When to use async:
- Web servers (FastAPI/Starlette) handling multiple requests
- Agents/pipelines doing parallel searches
- Concurrent document processing
- Notebooks where you want non-blocking operations

For simple scripts, sync methods are fine.
"""

import asyncio
import time

from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.zeusdb import ZeusDBVectorStore

load_dotenv()

# Configure OpenAI
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

print("=" * 70)
print("ZeusDB Async Examples")
print("=" * 70)
print()


# =============================================================================
# Example 1: Basic Async Operations
# =============================================================================
async def example_basic_async():
    """Demonstrate basic async add, query, and delete operations."""
    print("=" * 70)
    print("Example 1: Basic Async Operations")
    print("=" * 70)
    print()

    # Create vector store
    vector_store = ZeusDBVectorStore(dim=1536, distance="cosine", index_type="hnsw")

    # Create sample nodes
    nodes = [
        TextNode(
            text=f"Document {i}: Sample content about technology.",
            metadata={"doc_id": i, "category": "tech"},
        )
        for i in range(10)
    ]

    # Generate embeddings
    print("Generating embeddings...")
    embed_model = Settings.embed_model
    for node in nodes:
        node.embedding = embed_model.get_text_embedding(node.text)

    # Async add
    print(f"Adding {len(nodes)} nodes asynchronously...")
    start = time.perf_counter()
    # node_ids = await vector_store.aadd(nodes)
    node_ids = await vector_store.async_add(nodes)
    add_time = (time.perf_counter() - start) * 1000
    print(f"  ✅ Added {len(node_ids)} nodes in {add_time:.2f}ms")
    print()

    # Async query
    print("Querying asynchronously...")
    query_text = "technology"
    query_embedding = embed_model.get_text_embedding(query_text)
    query_obj = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=3)

    start = time.perf_counter()
    results = await vector_store.aquery(query_obj)
    query_time = (time.perf_counter() - start) * 1000

    result_ids = results.ids or []
    result_sims = results.similarities or []

    print(f"  ✅ Query completed in {query_time:.2f}ms")
    print(f"  Found {len(result_ids)} results:")
    for i, (node_id, similarity) in enumerate(zip(result_ids, result_sims), 1):
        print(f"    {i}. Score: {similarity:.4f}, ID: {node_id}")
    print()

    # Async delete
    print("Deleting nodes asynchronously...")
    start = time.perf_counter()
    await vector_store.adelete_nodes(node_ids=node_ids[:3])
    delete_time = (time.perf_counter() - start) * 1000
    print(f"  ✅ Deleted 3 nodes in {delete_time:.2f}ms")
    print(f"  Remaining nodes: {vector_store.get_vector_count()}")
    print()

    print(f"✅ Example 1 completed. Final count: {vector_store.get_vector_count()}")
    print()


# =============================================================================
# Example 2: Concurrent Queries
# =============================================================================
async def example_concurrent_queries():
    """Demonstrate running multiple queries concurrently."""
    print("=" * 70)
    print("Example 2: Concurrent Queries")
    print("=" * 70)
    print()

    # Setup
    vector_store = ZeusDBVectorStore(dim=1536, distance="cosine")
    embed_model = Settings.embed_model

    # Create diverse documents
    documents = [
        "Python is a programming language for data science and web development.",
        "Machine learning models require training data and compute resources.",
        "Climate change is affecting global weather patterns and ecosystems.",
        "Quantum computing uses qubits for parallel computation.",
        "Electric vehicles are becoming more popular due to environmental concerns.",
        "Artificial intelligence can analyze large datasets efficiently.",
        "Renewable energy sources include solar, wind, and hydroelectric power.",
        "Neural networks are inspired by biological brain structures.",
    ]

    nodes = []
    print("Preparing documents...")
    for i, text in enumerate(documents):
        node = TextNode(text=text, metadata={"doc_id": i, "source": "examples"})
        node.embedding = embed_model.get_text_embedding(text)
        nodes.append(node)

    # await vector_store.aadd(nodes)
    await vector_store.async_add(nodes)
    print(f"  ✅ Added {len(nodes)} documents")
    print()

    # Define multiple queries
    queries = [
        "programming languages",
        "machine learning",
        "climate and environment",
        "quantum computing",
    ]

    print(f"Running {len(queries)} queries concurrently...")
    print()

    # Sequential version (for comparison)
    print("📊 Sequential execution:")
    start = time.perf_counter()
    seq_results = []
    for query_text in queries:
        query_embedding = embed_model.get_text_embedding(query_text)
        query_obj = VectorStoreQuery(
            query_embedding=query_embedding, similarity_top_k=2
        )
        result = await vector_store.aquery(query_obj)
        seq_results.append(result)
    seq_time = (time.perf_counter() - start) * 1000
    print(f"  ⏱️  Total time: {seq_time:.2f}ms")
    print()

    # Concurrent version
    print("⚡ Concurrent execution:")

    async def query_single(query_text: str):
        query_embedding = embed_model.get_text_embedding(query_text)
        query_obj = VectorStoreQuery(
            query_embedding=query_embedding, similarity_top_k=2
        )
        return await vector_store.aquery(query_obj)

    start = time.perf_counter()
    concurrent_results = await asyncio.gather(*[query_single(q) for q in queries])
    concurrent_time = (time.perf_counter() - start) * 1000

    print(f"  ⏱️  Total time: {concurrent_time:.2f}ms")
    speedup = seq_time / concurrent_time if concurrent_time > 0 else 0
    print(f"  🚀 Speedup: {speedup:.2f}x faster")
    print()

    # Display results
    print("Results:")
    for query_text, result in zip(queries, concurrent_results):
        result_ids = result.ids or []
        result_sims = result.similarities or []
        print(f"  Query: '{query_text}'")
        for i, (node_id, sim) in enumerate(zip(result_ids, result_sims), 1):
            print(f"    {i}. Score: {sim:.4f}")
        print()

    print(f"✅ Example 2 completed. Total documents: {vector_store.get_vector_count()}")
    print()


# =============================================================================
# Example 3: Concurrent Document Processing
# =============================================================================
async def example_concurrent_processing():
    """Demonstrate processing multiple document batches concurrently."""
    print("=" * 70)
    print("Example 3: Concurrent Document Processing")
    print("=" * 70)
    print()

    vector_store = ZeusDBVectorStore(dim=1536, distance="cosine")
    embed_model = Settings.embed_model

    # Simulate multiple batches of documents
    batches = [
        [f"Tech document {i}" for i in range(5)],
        [f"Science document {i}" for i in range(5)],
        [f"Business document {i}" for i in range(5)],
    ]

    print(f"Processing {len(batches)} batches concurrently...")
    print()

    async def process_batch(batch: list[str], batch_id: int):
        """Process a single batch of documents."""
        start = time.perf_counter()

        nodes = []
        for i, text in enumerate(batch):
            node = TextNode(
                text=text,
                metadata={
                    "batch_id": batch_id,
                    "doc_id": i,
                },
            )
            node.embedding = embed_model.get_text_embedding(text)
            nodes.append(node)

        # node_ids = await vector_store.aadd(nodes)
        node_ids = await vector_store.async_add(nodes)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"  ✅ Batch {batch_id}: Added {len(node_ids)} nodes in {elapsed:.2f}ms")
        return node_ids

    # Process all batches concurrently
    start = time.perf_counter()
    all_ids = await asyncio.gather(
        *[process_batch(batch, i) for i, batch in enumerate(batches)]
    )
    total_time = (time.perf_counter() - start) * 1000

    total_docs = sum(len(ids) for ids in all_ids)
    print()
    print("📊 Summary:")
    print(f"  Total documents: {total_docs}")
    print(f"  Total time: {total_time:.2f}ms")
    print(f"  Avg per document: {total_time / total_docs:.2f}ms")
    print()

    print(f"✅ Example 3 completed. Total documents: {vector_store.get_vector_count()}")
    print()


# =============================================================================
# Example 4: Async with Error Handling
# =============================================================================
async def example_error_handling():
    """Demonstrate proper error handling in async operations."""
    print("=" * 70)
    print("Example 4: Async Error Handling")
    print("=" * 70)
    print()

    vector_store = ZeusDBVectorStore(dim=1536, distance="cosine")
    embed_model = Settings.embed_model

    # Add some documents
    nodes = []
    for i in range(5):
        node = TextNode(text=f"Document {i}", metadata={"doc_id": i})
        node.embedding = embed_model.get_text_embedding(node.text)
        nodes.append(node)

    # node_ids = await vector_store.aadd(nodes)
    node_ids = await vector_store.async_add(nodes)
    print(f"Added {len(node_ids)} documents")
    print()

    # Demonstrate handling of unsupported operation
    print("Testing unsupported delete by ref_doc_id...")
    try:
        await vector_store.adelete(ref_doc_id="some_doc")
        print("  ❌ Should have raised NotImplementedError")
    except NotImplementedError as e:
        print(f"  ✅ Correctly caught error: {str(e)[:60]}...")
    print()

    # Test unsupported clear operation
    print("Testing unsupported clear operation...")
    try:
        await vector_store.aclear()
        print("  ❌ Should have raised NotImplementedError")
    except NotImplementedError as e:
        print(f"  ✅ Correctly caught error: {str(e)[:50]}...")
    print()

    # Demonstrate successful delete by node IDs
    print("Testing supported delete by node IDs...")
    try:
        await vector_store.adelete_nodes(node_ids=node_ids[:2])
        print("  ✅ Successfully deleted 2 nodes")
        print(f"  Remaining: {vector_store.get_vector_count()}")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
    print()

    # Demonstrate concurrent operations with error handling
    print("Testing concurrent operations with error handling...")

    async def safe_query(query_text: str, query_id: int):
        """Query with error handling."""
        try:
            query_embedding = embed_model.get_text_embedding(query_text)
            query_obj = VectorStoreQuery(
                query_embedding=query_embedding, similarity_top_k=2
            )
            result = await vector_store.aquery(query_obj)
            return {"id": query_id, "success": True, "result": result}
        except Exception as e:
            return {"id": query_id, "success": False, "error": str(e)}

    queries = ["technology", "science", "business"]
    results = await asyncio.gather(
        *[safe_query(q, i) for i, q in enumerate(queries)], return_exceptions=True
    )

    success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    print(f"  ✅ {success_count}/{len(queries)} queries succeeded")
    print()

    print(f"✅ Example 4 completed. Final count: {vector_store.get_vector_count()}")
    print()


# =============================================================================
# Example 5: Async with Timeouts
# =============================================================================
async def example_timeouts():
    """Demonstrate using timeouts with async operations."""
    print("=" * 70)
    print("Example 5: Async Operations with Timeouts")
    print("=" * 70)
    print()

    vector_store = ZeusDBVectorStore(dim=1536, distance="cosine")
    embed_model = Settings.embed_model

    # Add documents
    nodes = []
    for i in range(10):
        node = TextNode(
            text=f"Sample document {i} about various topics.", metadata={"doc_id": i}
        )
        node.embedding = embed_model.get_text_embedding(node.text)
        nodes.append(node)

    print("Adding documents with timeout...")
    try:
        node_ids = await asyncio.wait_for(
            # vector_store.aadd(nodes),
            vector_store.async_add(nodes),
            timeout=10.0,  # 10 second timeout
        )
        print(f"  ✅ Added {len(node_ids)} nodes within timeout")
    except asyncio.TimeoutError:
        print("  ❌ Operation timed out")
    print()

    # Query with timeout
    print("Querying with timeout...")
    query_text = "sample topics"
    query_embedding = embed_model.get_text_embedding(query_text)
    query_obj = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=3)

    try:
        result = await asyncio.wait_for(
            vector_store.aquery(query_obj),
            timeout=5.0,  # 5 second timeout
        )
        result_ids = result.ids or []
        print(f"  ✅ Query returned {len(result_ids)} results within timeout")
    except asyncio.TimeoutError:
        print("  ❌ Query timed out")
    print()

    print(f"✅ Example 5 completed. Final count: {vector_store.get_vector_count()}")
    print()


# =============================================================================
# Main Execution
# =============================================================================
async def main():
    """Run all async examples."""
    try:
        await example_basic_async()
        await example_concurrent_queries()
        await example_concurrent_processing()
        await example_error_handling()
        await example_timeouts()

        print("=" * 70)
        print("Summary: Async Benefits")
        print("=" * 70)
        print()
        print("✅ Non-blocking operations - don't freeze your application")
        print("✅ Concurrent execution - run multiple operations simultaneously")
        print("✅ Better resource utilization - efficient I/O handling")
        print("✅ Improved throughput - process more requests per second")
        print("✅ Timeout support - prevent operations from hanging")
        print()
        print("💡 When to use async:")
        print("  • Web servers handling multiple requests (FastAPI, Starlette)")
        print("  • Agents/pipelines with parallel searches")
        print("  • Concurrent document processing")
        print("  • Notebooks with non-blocking operations")
        print()
        print("💡 When sync is fine:")
        print("  • Simple scripts with sequential operations")
        print("  • Single-threaded batch processing")
        print("  • Quick prototypes and experiments")
        print()
        print("📚 Available async methods:")
        # print("  • aadd(nodes) - Add nodes asynchronously")
        print("  • async_add(nodes) - Add nodes asynchronously")
        print("  • aquery(query) - Query asynchronously")
        print("  • adelete_nodes(node_ids) - Delete by IDs asynchronously")
        print()
        print("⚠️  Note: aclear() not yet implemented in ZeusDB backend")
        print()

    except Exception as e:
        print(f"❌ Error in examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # For running as a script
    asyncio.run(main())
