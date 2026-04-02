# Architecture Proposal: High-Performance Parallel Ingestion for LlamaIndex

## Current Issues

- Sequential transformation bottleneck for CPU-bound tasks (parsing, chunking).
- Unstable multiprocessing implementation leading to `PicklingError`.
- Lack of vectorized data processing for node metadata.

## Proposed Solutions

### 1. Vectorized Metadata Management (Polars Integration)

- Convert internal Node sequences to **Polars DataFrames** during the ingestion phase.
- Leverage Polars' multi-threaded expressions for metadata extraction and filtering.

### 2. Robust Parallelism (Enhanced num_workers)

- Refactor `ProcessPoolExecutor` usage to use a `spawn` context by default, preventing deadlocks with CUDA/GPU drivers.
- Implement a `SafePickle` wrapper for custom transformations.

### 3. Balanced Load Distribution

- Replace static batching with **Token-Aware Batching**: workers receive batches with roughly equal token counts to prevent "straggler" processes.

## Impact

- Expected **3x-5x speedup** for local document parsing.
- Improved stability for enterprise-scale RAG pipelines.
