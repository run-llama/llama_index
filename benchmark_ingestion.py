import time
import asyncio
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter

# 1. Prepare Data
# Larger set to overcome orchestration overhead
docs = [Document(text="word " * 1000) for _ in range(100)] + \
       [Document(text="word " * 10) for _ in range(1000)]

pipeline = IngestionPipeline(
    transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=20)]
)

def run_benchmark(num_workers=None):
    print(f"\n--- Benchmarking Ingestion (num_workers={num_workers}) ---")
    start_time = time.time()
    nodes = pipeline.run(documents=docs, num_workers=num_workers)
    end_time = time.time()
    print(f"Nodes created: {len(nodes)}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    # Test sequential
    run_benchmark(num_workers=None)
    # Test current parallel
    run_benchmark(num_workers=4)
