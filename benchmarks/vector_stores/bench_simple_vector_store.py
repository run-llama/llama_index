import random
import time
from typing import List

from llama_index.schema import TextNode
from llama_index.vector_stores.simple import SimpleVectorStore
from llama_index.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)


def generate_nodes(
    num_vectors: int = 100, embedding_length: int = 1536
) -> List[TextNode]:
    random.seed(42)  # Make this reproducible
    return [
        TextNode(
            embedding=[random.uniform(0, 1) for _ in range(embedding_length)],
        )
        for _ in range(num_vectors)
    ]


def bench_simple_vector_store(
    num_vectors: List[int] = [10, 50, 100, 500, 1000],
) -> None:
    """Benchmark simple vector store."""
    print("Benchmarking SimpleVectorStore\n---------------------------")
    for num_vector in num_vectors:
        nodes = generate_nodes(num_vectors=num_vector)

        vector_store = SimpleVectorStore()

        time1 = time.time()
        vector_store.add(nodes=nodes)
        time2 = time.time()
        print(f"Adding {num_vector} vectors took {time2 - time1} seconds")

        for mode in [
            VectorStoreQueryMode.DEFAULT,
            VectorStoreQueryMode.SVM,
            VectorStoreQueryMode.MMR,
        ]:
            time1 = time.time()
            query = VectorStoreQuery(
                query_embedding=nodes[0].get_embedding(), similarity_top_k=10, mode=mode
            )
            vector_store.query(query=query)
            time2 = time.time()
            print(
                f"""Querying store of {num_vector} \
vectors with {mode} mode took {time2 - time1} seconds"""
            )


if __name__ == "__main__":
    bench_simple_vector_store()
