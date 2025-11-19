"""
MongoDB Atlas Vector & Hybrid Search latency benchmark.

Run this script to capture approximate latencies of different query modes after the performance fix
(server-side filtering + removal of empty filter dict).

Usage (from repo root):
    python llama-index-integrations/vector_stores/llama-index-vector-stores-mongodb/scripts/benchmark_mongodb_search.py --docs 100000

Requirements:
    - Environment variable MONGODB_URI (Atlas cluster >= M10 with vector & search enabled)
    - No API keys required (uses synthetic embeddings by default)

Outputs:
    JSON object with per-mode statistics (min/median/mean/max) over N runs.

Notes:
    This is a lightweight benchmark; results are indicative, not authoritative.
    Network conditions and Atlas load will affect times. For comparative analysis,
    run before and after code changes and diff the JSON outputs.

    Uses synthetic random embeddings (1536 dimensions) for reproducible benchmarks
    without API costs. Performance tests MongoDB vector search, not embedding quality.

    Tests both unfiltered and filtered variants of vector, text, and hybrid searches
    using high-selectivity filters to demonstrate pre-filtering benefits.

"""

from __future__ import annotations

import json
import os
import statistics
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, cast

from pymongo import MongoClient

from llama_index.core.schema import BaseNode, Document, TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
# Fallback for running script from monorepo root where integration distribution isn't installed.
try:
    from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch  # type: ignore
except ModuleNotFoundError:
    # We are in the monorepo; ascend until we reach the integration package root (contains pyproject.toml
    # for llama-index-vector-stores-mongodb) then add that directory to sys.path.
    import sys
    import os

    cur = os.path.dirname(__file__)  # .../scripts
    target_pyproject = "pyproject.toml"
    package_root = None
    for _ in range(4):  # climb up to 4 levels max
        candidate = os.path.abspath(cur)
        if os.path.isfile(os.path.join(candidate, target_pyproject)) and "vector-stores-mongodb" in candidate:
            package_root = candidate
            break
        cur = os.path.join(cur, "..")

    if package_root and package_root not in sys.path:
        sys.path.insert(0, package_root)

    # Add monorepo core package path (/_llama-index) if present so embeddings and core modules resolve.
    # Ascend to repository root first.
    repo_root = None
    probe = package_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for _ in range(6):
        candidate = os.path.abspath(probe)
        if os.path.isdir(os.path.join(candidate, "_llama-index")):
            repo_root = candidate
            break
        probe = os.path.join(probe, "..")
    if repo_root:
        core_path = os.path.join(repo_root, "_llama-index")
        if core_path not in sys.path:
            sys.path.insert(0, core_path)
    # Final fallback: one level up from scripts (integration root) in case detection failed.
    integration_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if integration_root not in sys.path:
        sys.path.insert(0, integration_root)

    try:
        from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"Unable to import MongoDBAtlasVectorSearch; checked package_root={package_root}; sys.path={sys.path}\nOriginal error: {e}"
        )


MONGODB_URI = os.environ.get("MONGODB_URI")
DB_NAME = os.environ.get("MONGODB_DATABASE", "llama_index_test_db")
COLL_NAME = os.environ.get("MONGODB_COLLECTION", "llama_index_benchmark_collection")
VECTOR_INDEX_NAME = os.environ.get("MONGODB_INDEX", "vector_index")
FULLTEXT_INDEX_NAME = "fulltext_index"
RUNS = int(os.environ.get("MONGODB_BENCH_RUNS", 5))
WARMUP = int(os.environ.get("MONGODB_BENCH_WARMUP", 2))
TARGET_DOCS = int(os.environ.get("MONGODB_BENCH_TARGET_DOCS", 100))  # scalable corpus size
EXPORT_PATH = os.environ.get("MONGODB_BENCH_OUT")


@dataclass
class TimingStats:
    label: str
    runs: int
    min_ms: float
    median_ms: float
    mean_ms: float
    max_ms: float
    stddev_ms: float
    p90_ms: float
    p95_ms: float


def _ensure_indexes(vs: MongoDBAtlasVectorSearch, dims: int) -> None:
    existing = {idx["name"] for idx in vs.collection.list_search_indexes()}
    # Always ensure filter path present (update if index already exists)
    if VECTOR_INDEX_NAME not in existing:
        vs.create_vector_search_index(
            path=vs._embedding_key,
            dimensions=dims,
            similarity="cosine",
            filters=["metadata.text"],
            wait_until_complete=180,
        )
    else:
        # Force update to guarantee filter path included (idempotent if already there)
        try:
            vs.update_vector_search_index(
                path=vs._embedding_key,
                dimensions=dims,
                similarity="cosine",
                filters=["metadata.text"],
                wait_until_complete=180,
            )
        except Exception:
            pass  # Non-fatal; proceed (benchmark will fail later if truly unusable)
    if FULLTEXT_INDEX_NAME not in existing:
        # Create fulltext index with both the main text field and metadata.text field
        # This is needed for filtered text search queries
        from pymongo.operations import SearchIndexModel
        definition = {
            "mappings": {
                "dynamic": False,
                "fields": {
                    vs._text_key: {"type": "string"},
                    "metadata.text": {"type": "token"}
                }
            }
        }
        vs._collection.create_search_index(
            SearchIndexModel(
                definition=definition,
                name=FULLTEXT_INDEX_NAME,
                type="search",
            )
        )
        # Wait for index to be ready
        import time
        timeout = 180
        start_time = time.time()
        while time.time() - start_time < timeout:
            indexes = {idx["name"]: idx for idx in vs._collection.list_search_indexes()}
            if FULLTEXT_INDEX_NAME in indexes:
                status = indexes[FULLTEXT_INDEX_NAME].get("status")
                if status == "READY":
                    break
            time.sleep(2)
        else:
            print(f"Warning: Index {FULLTEXT_INDEX_NAME} did not become READY within {timeout}s")
    else:
        # Index exists - check if it has the correct definition with metadata.text
        # If not, drop and recreate it
        from pymongo.operations import SearchIndexModel
        indexes = {idx["name"]: idx for idx in vs._collection.list_search_indexes()}
        current_def = indexes.get(FULLTEXT_INDEX_NAME, {})
        
        # Check if metadata.text is in the index definition
        needs_update = True
        if "latestDefinition" in current_def:
            mappings = current_def.get("latestDefinition", {}).get("mappings", {})
            fields = mappings.get("fields", {})
            if "metadata.text" in fields:
                needs_update = False
        
        if needs_update:
            print(f"Updating {FULLTEXT_INDEX_NAME} to include metadata.text field...")
            # Drop the old index
            vs._collection.drop_search_index(FULLTEXT_INDEX_NAME)
            
            # Wait for deletion to complete
            import time
            timeout = 60
            start_time = time.time()
            while time.time() - start_time < timeout:
                existing_names = {idx["name"] for idx in vs._collection.list_search_indexes()}
                if FULLTEXT_INDEX_NAME not in existing_names:
                    break
                time.sleep(2)
            
            # Create new index with correct definition
            definition = {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        vs._text_key: {"type": "string"},
                        "metadata.text": {"type": "token"}
                    }
                }
            }
            vs._collection.create_search_index(
                SearchIndexModel(
                    definition=definition,
                    name=FULLTEXT_INDEX_NAME,
                    type="search",
                )
            )
            
            # Wait for index to be ready
            timeout = 180
            start_time = time.time()
            while time.time() - start_time < timeout:
                indexes = {idx["name"]: idx for idx in vs._collection.list_search_indexes()}
                if FULLTEXT_INDEX_NAME in indexes:
                    status = indexes[FULLTEXT_INDEX_NAME].get("status")
                    if status == "READY":
                        break
                time.sleep(2)
            else:
                print(f"Warning: Index {FULLTEXT_INDEX_NAME} did not become READY within {timeout}s")


def _prepare_collection(
    vs: MongoDBAtlasVectorSearch,
    target_docs: int,
) -> List[Document]:
    """
    Populate the collection with synthetic embeddings.

    Uses random 1536-dimensional vectors for fast, reproducible benchmarks
    without API costs. Always reuses existing documents and only adds more
    if needed to reach target_docs. Never deletes existing documents.
    """
    # Use estimated_document_count() which is much faster than count_documents()
    # for large collections (uses metadata instead of scanning)
    existing_count = vs._collection.estimated_document_count()
    sample_docs: List[Document] = []

    if existing_count >= target_docs:
        # Already have enough documents
        print(f"Found ~{existing_count} existing documents (>= requested {target_docs}). Using existing data.")
        cursor = vs._collection.find({}, {vs._text_key: 1, "metadata": 1}).limit(min(25, existing_count))
        for doc in cursor:
            sample_docs.append(
                Document(text=doc.get(vs._text_key, ""), metadata=doc.get("metadata", {}))
            )
        if sample_docs:
            return sample_docs
        else:
            raise SystemExit("Failed to retrieve sample documents from collection")

    # Need to add more documents to reach target
    docs_to_add = target_docs - existing_count
    print(f"Found ~{existing_count} existing documents. Adding {docs_to_add} more to reach {target_docs}...")

    base_lines = Document.example().text.split("\n")[:25]

    # Generate synthetic embeddings (random 1536-dimensional vectors)
    import random
    dims = 1536

    BATCH_SIZE = 10000  # Process 10k documents at a time

    print(f"[{time.strftime('%H:%M:%S')}] Inserting {docs_to_add} documents in batches of {BATCH_SIZE}...")
    total_inserted = 0
    batch_num = 0

    while total_inserted < docs_to_add:
        batch_num += 1
        batch_start = time.perf_counter()
        batch_size = min(BATCH_SIZE, docs_to_add - total_inserted)
        nodes: List[TextNode] = []
        docs: List[Document] = []

        # Generate batch of documents
        gen_start = time.perf_counter()
        for i in range(batch_size):
            doc_idx = existing_count + total_inserted + i
            line_idx = doc_idx % len(base_lines)
            cycle = doc_idx // len(base_lines)
            line = base_lines[line_idx]
            suffix = f" #{cycle}" if cycle > 0 else ""
            txt = f"{line}{suffix} llamaindex LLM"
            doc = Document(text=txt, metadata={"text": line, "group": doc_idx % 5})
            docs.append(doc)

            # Create synthetic embedding
            embedding = [random.uniform(-1.0, 1.0) for _ in range(dims)]
            nodes.append(
                TextNode(text=doc.text, embedding=embedding, metadata=doc.metadata, id_=str(doc_idx))
            )
        gen_time = time.perf_counter() - gen_start

        # Insert batch via library method (already uses insert_many internally)
        # Note: Slowness is due to MongoDB vector index updates, not Python overhead
        insert_start = time.perf_counter()
        vs.add(cast(List[BaseNode], nodes))
        insert_time = time.perf_counter() - insert_start
        total_inserted += batch_size
        batch_time = time.perf_counter() - batch_start
        print(f"  [{time.strftime('%H:%M:%S')}] Batch {batch_num}: Inserted {batch_size} documents (total: {existing_count + total_inserted}/{target_docs}) - batch: {batch_time:.1f}s (gen: {gen_time:.1f}s, insert: {insert_time:.1f}s)")

        # Keep a sample of documents for query generation (from first batch only)
        if batch_num == 1:
            sample_docs = docs[:min(25, len(docs))]

    # If we didn't insert anything (existing_count >= target_docs), get sample from existing docs
    if not sample_docs:
        cursor = vs._collection.find({}, {vs._text_key: 1, "metadata": 1}).limit(25)
        for doc in cursor:
            sample_docs.append(
                Document(text=doc.get(vs._text_key, ""), metadata=doc.get("metadata", {}))
            )

    return sample_docs


def time_query(label: str, fn: Callable[[], None]) -> List[float]:
    # Warm-up
    for _ in range(WARMUP):
        fn()
    times: List[float] = []
    for _ in range(RUNS):
        start = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
    return times


def _percentile(times: List[float], pct: float) -> float:
    if not times:
        return 0.0
    sorted_times = sorted(times)
    # nearest rank
    k = max(1, int(round(pct * len(sorted_times))))
    return sorted_times[min(k - 1, len(sorted_times) - 1)]


def stats(label: str, times: List[float]) -> TimingStats:
    return TimingStats(
        label=label,
        runs=len(times),
        min_ms=min(times),
        median_ms=statistics.median(times),
        mean_ms=statistics.mean(times),
        max_ms=max(times),
        stddev_ms=statistics.pstdev(times) if len(times) > 1 else 0.0,
        p90_ms=_percentile(times, 0.90),
        p95_ms=_percentile(times, 0.95),
    )


def main(argv: Optional[List[str]] = None) -> None:
    if MONGODB_URI is None:
        raise SystemExit("MONGODB_URI is required")
    # Declare globals early since we reference them for argparse defaults.
    global RUNS, WARMUP
    parser = argparse.ArgumentParser(description="MongoDB Atlas Vector & Hybrid benchmark")
    parser.add_argument("--runs", type=int, default=RUNS, help="Timed runs per mode")
    parser.add_argument("--warmup", type=int, default=WARMUP, help="Warm-up iterations per mode")
    parser.add_argument("--docs", type=int, default=TARGET_DOCS, help="Target document count to ingest")
    parser.add_argument("--out", type=str, default=EXPORT_PATH, help="Optional JSON output file path")
    args = parser.parse_args(argv)

    # Override global counters for this invocation
    RUNS = args.runs
    WARMUP = args.warmup

    # Use synthetic embeddings (1536 dimensions) for benchmark
    dims = 1536

    client: MongoClient = MongoClient(MONGODB_URI)
    vs = MongoDBAtlasVectorSearch(
        mongodb_client=client,
        db_name=DB_NAME,
        collection_name=COLL_NAME,
        vector_index_name=VECTOR_INDEX_NAME,
        fulltext_index_name=FULLTEXT_INDEX_NAME,
    )

    t0 = time.perf_counter()
    _ensure_indexes(vs, dims=dims)
    index_ensure_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    docs = _prepare_collection(vs, target_docs=args.docs)
    ingest_ms = (time.perf_counter() - t1) * 1000
    print(
        json.dumps(
            {
                "setup": {
                    "docs": len(docs),
                    "index_ensure_ms": round(index_ensure_ms, 2),
                    "ingest_ms": round(ingest_ms, 2),
                }
            }
        )
    )

    # Pre-build embeddings & query objects
    vector_query_text = docs[0].text
    hybrid_text_query = "llamaindex"

    # Generate synthetic query embedding (random 1536-dimensional vector)
    import random
    embedding_vector = [random.uniform(-1.0, 1.0) for _ in range(dims)]

    # High selectivity filter: matches ~4% of documents (1/25 base_lines)
    # This demonstrates the performance benefits of server-side pre-filtering
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="text", value=docs[0].metadata["text"], operator=FilterOperator.EQ
            )
        ]
    )

    def vector_query() -> None:
        vs.query(
            VectorStoreQuery(
                query_embedding=embedding_vector,
                similarity_top_k=5,
                mode=VectorStoreQueryMode.DEFAULT,
            )
        )

    def filtered_vector_query() -> None:
        vs.query(
            VectorStoreQuery(
                query_embedding=embedding_vector,
                similarity_top_k=5,
                mode=VectorStoreQueryMode.DEFAULT,
                filters=filters,
            )
        )

    def text_query() -> None:
        vs.query(
            VectorStoreQuery(
                query_str=hybrid_text_query,
                similarity_top_k=5,
                mode=VectorStoreQueryMode.TEXT_SEARCH,
            )
        )

    def filtered_text_query() -> None:
        vs.query(
            VectorStoreQuery(
                query_str=hybrid_text_query,
                similarity_top_k=5,
                mode=VectorStoreQueryMode.TEXT_SEARCH,
                filters=filters,
            )
        )

    def hybrid_query() -> None:
        vs.query(
            VectorStoreQuery(
                query_embedding=embedding_vector,
                query_str=hybrid_text_query,
                similarity_top_k=5,
                mode=VectorStoreQueryMode.HYBRID,
                alpha=0.5,
            )
        )

    def filtered_hybrid_query() -> None:
        vs.query(
            VectorStoreQuery(
                query_embedding=embedding_vector,
                query_str=hybrid_text_query,
                similarity_top_k=5,
                mode=VectorStoreQueryMode.HYBRID,
                alpha=0.5,
                filters=filters,
            )
        )

    results: Dict[str, TimingStats] = {}
    results["vector"] = stats("vector", time_query("vector", vector_query))
    results["vector_filtered"] = stats(
        "vector_filtered", time_query("vector_filtered", filtered_vector_query)
    )
    results["text_search"] = stats("text_search", time_query("text_search", text_query))
    results["text_search_filtered"] = stats(
        "text_search_filtered", time_query("text_search_filtered", filtered_text_query)
    )
    results["hybrid"] = stats("hybrid", time_query("hybrid", hybrid_query))
    results["hybrid_filtered"] = stats(
        "hybrid_filtered", time_query("hybrid_filtered", filtered_hybrid_query)
    )

    summary = {k: asdict(v) for k, v in results.items()}
    print(json.dumps(summary, indent=2))
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Results written to {args.out}")
        except OSError as e:
            print(f"Failed to write results to {args.out}: {e}")


if __name__ == "__main__":
    main()
