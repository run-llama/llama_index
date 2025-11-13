"""
MongoDB Atlas Vector & Hybrid Search latency benchmark.

Run this script to capture approximate latencies of different query modes after the performance fix
(server-side filtering + removal of empty filter dict).

Usage (from repo root):
    uv run python llama-index-integrations/vector_stores/llama-index-vector-stores-mongodb/scripts/benchmark_mongodb_search.py

Requirements:
    - Environment variable MONGODB_URI (Atlas cluster >= M10 with vector & search enabled)
    - OPENAI_API_KEY or AZURE_OPENAI_API_KEY for embedding

Outputs:
    JSON object with per-mode statistics (min/median/mean/max) over N runs.

Notes:
    This is a lightweight benchmark; results are indicative, not authoritative.
    Network conditions and Atlas load will affect times. For comparative analysis,
    run before and after code changes and diff the JSON outputs.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

from pymongo import MongoClient

from llama_index.core import Settings
from llama_index.core.schema import Document, TextNode
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
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter


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


def _embed_model() -> None:
    if Settings.embed_model is not None:
        return
    # Lazy import with monorepo path fallback so embeddings integration packages resolve.
    import sys
    import os

    def _ensure_embedding_pkg(pkg_dir: str) -> None:
        # Find the embeddings package directory relative to this script.
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # up to vector_stores
        integrations_root = os.path.abspath(os.path.join(base, ".."))  # up to llama-index-integrations
        candidate = os.path.join(integrations_root, "embeddings", pkg_dir)
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)

    if "OPENAI_API_KEY" in os.environ:
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding  # type: ignore
        except ModuleNotFoundError:
            _ensure_embedding_pkg("llama-index-embeddings-openai")
            from llama_index.embeddings.openai import OpenAIEmbedding  # type: ignore
        Settings.embed_model = OpenAIEmbedding()
    elif "AZURE_OPENAI_API_KEY" in os.environ:
        deployment = os.environ.get("AZURE_TEXT_DEPLOYMENT", "text-embedding-3-small")
        try:
            from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding  # type: ignore
        except ModuleNotFoundError:
            _ensure_embedding_pkg("llama-index-embeddings-azure-openai")
            from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding  # type: ignore
        Settings.embed_model = AzureOpenAIEmbedding(
            api_key=os.environ["AZURE_OPENAI_API_KEY"], deployment_name=deployment
        )
    else:
        raise SystemExit("Requires OPENAI_API_KEY or AZURE_OPENAI_API_KEY for embeddings")


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
        vs.create_fulltext_search_index(
            field=vs._text_key,
            field_type="string",
            wait_until_complete=180,
        )


def _prepare_collection(
    vs: MongoDBAtlasVectorSearch,
    target_docs: int,
    synthetic: bool = False,
    reuse: bool = False,
    quick: bool = False,
) -> List[Document]:
    """
    Populate the collection.

    Modes:
        synthetic: generate random embeddings (1 embed model call for query + dim probe) instead of real model per doc.
        reuse: if existing doc count >= target_docs, skip ingestion entirely.
        quick: avoid sentence splitting; one node per document to reduce embedding calls.
    """
    if reuse:
        existing_count = vs._collection.count_documents({})
        if existing_count >= target_docs:
            # Build lightweight Document list from a sample of existing docs for queries.
            sample_docs: List[Document] = []
            cursor = vs._collection.find({}, {vs._text_key: 1, "metadata": 1}).limit(min(25, target_docs))
            for doc in cursor:
                sample_docs.append(
                    Document(text=doc.get(vs._text_key, ""), metadata=doc.get("metadata", {}))
                )
            if sample_docs:
                return sample_docs
            # Fallback to fresh ingest if sample retrieval failed.

    # Fresh ingestion path.
    vs._collection.delete_many({})
    time.sleep(1)  # small pause to let deletes settle
    base_lines = Document.example().text.split("\n")[:25]
    docs: List[Document] = []
    cycles = (target_docs + len(base_lines) - 1) // len(base_lines)
    for i in range(cycles):
        for line in base_lines:
            if len(docs) >= target_docs:
                break
            suffix = f" #{i}" if cycles > 1 else ""
            txt = f"{line}{suffix} llamaindex LLM"
            docs.append(Document(text=txt, metadata={"text": line, "group": i % 5}))
        if len(docs) >= target_docs:
            break

    # Ingestion strategies
    if synthetic:
        # Create random embeddings without calling the model per doc.
        import random
        dims = getattr(Settings.embed_model, "dimensions", None)
        if not dims:
            probe_vec = Settings.embed_model.get_text_embedding("dimension probe")
            dims = len(probe_vec)
        nodes: List[TextNode] = []
        for d in docs:
            embedding = [random.uniform(-1.0, 1.0) for _ in range(dims)]
            nodes.append(
                TextNode(text=d.text, embedding=embedding, metadata=d.metadata, id_=None)
            )
        vs.add(nodes)
        return docs
    else:
        # Real embedding path (can be quick or full).
        if quick:
            # One chunk per document; larger chunk size removes splitting overhead.
            pipeline = IngestionPipeline(
                transformations=[SentenceSplitter(chunk_size=8192, chunk_overlap=0), Settings.embed_model]
            )
        else:
            pipeline = IngestionPipeline(
                transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=200), Settings.embed_model]
            )
        nodes = pipeline.run(documents=docs)
        vs.add(nodes)
        return docs


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
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic random embeddings for ingestion (fast, non-semantic).",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing collection if it already has >= target docs (skip reingest).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick ingest: no splitting (one chunk per doc) to reduce embedding calls.",
    )
    args = parser.parse_args(argv)

    # Override global counters for this invocation
    RUNS = args.runs
    WARMUP = args.warmup

    _embed_model()
    client = MongoClient(MONGODB_URI)
    vs = MongoDBAtlasVectorSearch(
        mongodb_client=client,
        db_name=DB_NAME,
        collection_name=COLL_NAME,
        vector_index_name=VECTOR_INDEX_NAME,
        fulltext_index_name=FULLTEXT_INDEX_NAME,
    )
    dims = getattr(Settings.embed_model, "dimensions", None)
    if not dims:
        try:
            probe_vec = Settings.embed_model.get_text_embedding("dimension probe")
            dims = len(probe_vec)
        except Exception as e:  # pragma: no cover
            raise SystemExit(f"Failed to determine embedding dimensions: {e}")
    t0 = time.perf_counter()
    _ensure_indexes(vs, dims=dims)
    index_ensure_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    docs = _prepare_collection(
        vs, target_docs=args.docs, synthetic=args.synthetic, reuse=args.reuse, quick=args.quick
    )
    ingest_ms = (time.perf_counter() - t1) * 1000
    print(
        json.dumps(
            {
                "setup": {
                    "docs": len(docs),
                    "index_ensure_ms": round(index_ensure_ms, 2),
                    "ingest_ms": round(ingest_ms, 2),
                    "mode": {
                        "synthetic": args.synthetic,
                        "reuse": args.reuse,
                        "quick": args.quick,
                    },
                }
            }
        )
    )

    # Pre-build embeddings & query objects
    vector_query_text = docs[0].text
    hybrid_text_query = "llamaindex"
    embedding_vector = Settings.embed_model.get_text_embedding(vector_query_text)

    # Filter excluding first doc's text token to exercise filter server-side
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="text", value=docs[0].metadata["text"], operator=FilterOperator.NE
            )
        ]
    )

    def vector_query():
        vs.query(
            VectorStoreQuery(
                query_embedding=embedding_vector,
                similarity_top_k=5,
                mode=VectorStoreQueryMode.DEFAULT,
            )
        )

    def filtered_vector_query():
        vs.query(
            VectorStoreQuery(
                query_embedding=embedding_vector,
                similarity_top_k=5,
                mode=VectorStoreQueryMode.DEFAULT,
                filters=filters,
            )
        )

    def text_query():
        vs.query(
            VectorStoreQuery(
                query_str=hybrid_text_query,
                similarity_top_k=5,
                mode=VectorStoreQueryMode.TEXT_SEARCH,
            )
        )

    def hybrid_query():
        vs.query(
            VectorStoreQuery(
                query_embedding=embedding_vector,
                query_str=hybrid_text_query,
                similarity_top_k=5,
                mode=VectorStoreQueryMode.HYBRID,
                alpha=0.5,
            )
        )

    results: Dict[str, TimingStats] = {}
    results["vector"] = stats("vector", time_query("vector", vector_query))
    results["vector_filtered"] = stats(
        "vector_filtered", time_query("vector_filtered", filtered_vector_query)
    )
    results["text_search"] = stats("text_search", time_query("text_search", text_query))
    results["hybrid"] = stats("hybrid", time_query("hybrid", hybrid_query))

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
