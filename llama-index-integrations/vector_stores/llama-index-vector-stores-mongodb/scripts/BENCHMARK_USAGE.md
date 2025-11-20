# MongoDB Atlas Vector Search Benchmark Usage

## Overview

The benchmark script measures query latencies for MongoDB Atlas Vector Search and Hybrid Search operations. All queries are automatically tagged for easy identification in MongoDB Atlas.

## Basic Usage

```bash
# Run with default settings (100 documents)
python scripts/benchmark_mongodb_search.py

# Run with specific document count
python scripts/benchmark_mongodb_search.py --docs 100000

# Run with custom tag to identify queries
python scripts/benchmark_mongodb_search.py --docs 100000 --tag before_fix

# Run comparison benchmarks
python scripts/benchmark_mongodb_search.py --docs 100000 --tag before_fix --out before_fix.json
python scripts/benchmark_mongodb_search.py --docs 100000 --tag after_fix --out after_fix.json

# Use local source code instead of installed package
python scripts/benchmark_mongodb_search.py --docs 100000 --use-local --out local_version.json
```

## Command-Line Options

- `--docs`: Number of documents to use for benchmarking (default: 100)
- `--runs`: Number of timed runs per query mode (default: 5)
- `--warmup`: Number of warm-up iterations per mode (default: 2)
- `--tag`: Custom tag to identify queries in MongoDB Atlas (default: "benchmark")
- `--out`: Path to save JSON results file
- `--use-local`: Use local source code instead of installed package (for testing changes before release)

## Query Identification in MongoDB Atlas

### How It Works

Every benchmark query is automatically tagged with a comment that includes:

- The benchmark identifier: `llama_index_benchmark`
- Your custom tag: `tag=<your_tag>`

### Viewing Tagged Queries in Atlas

1. **Performance Advisor**:

   - Navigate to: Database → Performance Advisor
   - Look for queries with comment: `llama_index_benchmark tag=<your_tag>`

2. **Query Profiler**:

   - Navigate to: Database → Query Profiler
   - Filter by comment to see specific benchmark runs
   - Example search: `llama_index_benchmark tag=before_fix`

3. **Real-Time Performance Panel**:
   - Navigate to: Database → Real-Time Performance
   - View active queries with your tag during benchmark execution

### Use Cases

**Before/After Comparisons**:

```bash
# Run baseline with installed package
python scripts/benchmark_mongodb_search.py --docs 220000 --tag baseline --out baseline.json

# Make code changes...

# Run with local changes to test improvements
python scripts/benchmark_mongodb_search.py --docs 220000 --tag optimized --out optimized.json --use-local
```

**Comparing Installed vs Local Versions**:

```bash
# Test current installed version
python scripts/benchmark_mongodb_search.py --docs 100000 --out installed.json

# Test local development version
python scripts/benchmark_mongodb_search.py --docs 100000 --out local.json --use-local
```

**Testing Different Configurations**:

```bash
python scripts/benchmark_mongodb_search.py --tag config_a --runs 10
python scripts/benchmark_mongodb_search.py --tag config_b --runs 10
```

**Environment Testing**:

```bash
python scripts/benchmark_mongodb_search.py --tag dev_cluster
python scripts/benchmark_mongodb_search.py --tag prod_cluster
```

## Output Format

The script outputs JSON with timing statistics and your tag:

```json
{
  "tag": "before_fix",
  "results": {
    "vector": {
      "label": "vector",
      "runs": 5,
      "min_ms": 123.45,
      "median_ms": 145.67,
      "mean_ms": 150.23,
      "max_ms": 178.90,
      "stddev_ms": 20.15,
      "p90_ms": 170.12,
      "p95_ms": 175.43
    },
    "vector_filtered": { ... },
    "text_search": { ... },
    "hybrid": { ... }
  }
}
```

## Requirements

- **MongoDB URI**: Set via `MONGODB_URI` environment variable
- **Atlas Cluster**: M10 or higher with Vector Search enabled
- **Indexes**: Vector and full-text search indexes (auto-created if missing)

## Tips

- Use consistent `--docs` values when comparing results
- Run multiple iterations (`--runs`) for more reliable statistics
- Use descriptive tags that clearly identify the test scenario
- Save outputs to JSON files for later analysis and comparison
- Use `--use-local` when testing code changes before they are installed/released
- The script prints which version (local or installed) is being used at startup
