import random
import time
from typing import List
from llama_index.schema import TextNode

from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.simple import SimpleVectorStore
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import NodeParser
from llama_index.embeddings import OpenAIEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext


def generate_strings(
    num_strings: int = 100, string_length: int = 10
) -> List[NodeWithEmbedding]:
    content = (
        SimpleDirectoryReader("../../examples/paul_graham_essay/data")
        .load_data()[0]
        .get_content()
    )
    content_length = len(content)

    strings_per_loop = content_length / string_length
    strings = []

    for offset in range(0, int(num_strings / strings_per_loop) + 1):
        ptr = offset
        while ptr + string_length < content_length:
            strings.append(content[ptr : ptr + string_length])
            ptr += string_length
            if len(strings) == num_strings:
                break

    return strings


def bench_simple_vector_store(
    num_strings=[100],
    string_lengths=[128, 256, 512, 1024],
    embed_batch_sizes=[1, 10],
    torch_num_threads = None,
) -> None:
    """Benchmark embeddings."""
    print("Benchmarking Embeddings\n---------------------------")

    if torch_num_threads is not None:
        import torch
        torch.set_num_threads(torch_num_threads)

    max_num_strings = max(num_strings)
    for string_length in string_lengths:
        generated_strings = generate_strings(
            num_strings=max_num_strings, string_length=max(string_lengths)
        )

        for string_count in num_strings:
            strings = generated_strings[:string_count]

            for batch_size in embed_batch_sizes:
                embed_models = [
                    OpenAIEmbedding(embed_batch_size=batch_size),
                    LangchainEmbedding(
                        HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-mpnet-base-v2"
                        ),
                        embed_batch_size=batch_size,
                    ),
                    LangchainEmbedding(
                        HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2"
                        ),
                        embed_batch_size=batch_size,
                    ),
                ]

                embed_model_info = [
                    ("OpenAIEmbedding", 4096,),
                    (
                        "hf/sentence-transformers/all-mpnet-base-v2", 
                        embed_models[1]._langchain_embedding.client.max_seq_length,
                    ),
                    (
                        "hf/sentence-transformers/all-MiniLM-L6-v2",
                        embed_models[2]._langchain_embedding.client.max_seq_length,
                    ),
                ]

                skip_set = [0, 1] # skip openai

                for idx, model in enumerate(embed_models):
                    if idx in skip_set:
                        continue
                    for i, string in enumerate(strings):
                        model.queue_text_for_embedding(str(i), string)

                    time1 = time.time()
                    _ = model.get_queued_text_embeddings(show_progress=True)

                    time2 = time.time()
                    print(
                        f"""Embedding with model {embed_model_info[idx][0]} with batch size {batch_size} \
and max_seq_length {embed_model_info[idx][1]} for {string_count} strings of length {string_length} and \
took {time2 - time1} seconds"""
                    )
                # TODO: async version


if __name__ == "__main__":
    bench_simple_vector_store(torch_num_threads=12)
