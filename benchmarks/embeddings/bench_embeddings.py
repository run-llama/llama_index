import time
from typing import List, Optional, Tuple, Callable

from llama_index import SimpleDirectoryReader
from llama_index.embeddings import OpenAIEmbedding
from llama_index.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding
from functools import partial


def generate_strings(num_strings: int = 100, string_length: int = 10) -> List[str]:
    """
    Generate random strings sliced from the paul graham essay of the following form:

    offset 0: [0:string_length], [string_length:2*string_length], ...
    offset 1: [1:1+string_length], [1+string_length:1+2*string_length],...
    ...
    """
    content = (
        SimpleDirectoryReader("../../examples/paul_graham_essay/data")
        .load_data()[0]
        .get_content()
    )
    content_length = len(content)

    strings_per_loop = content_length / string_length
    num_loops_upper_bound = int(num_strings / strings_per_loop) + 1
    strings = []

    for offset in range(0, num_loops_upper_bound + 1):
        ptr = offset % string_length
        while ptr + string_length < content_length:
            strings.append(content[ptr : ptr + string_length])
            ptr += string_length
            if len(strings) == num_strings:
                break

    return strings


def create_open_ai_embedding(batch_size: int) -> Tuple[BaseEmbedding, str, int]:
    return (
        OpenAIEmbedding(embed_batch_size=batch_size),
        "OpenAIEmbedding",
        4096,
    )


def create_hf_embedding(
    model_name: str, batch_size: int
) -> Tuple[BaseEmbedding, str, int]:
    model = LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name=model_name,
        ),
        embed_batch_size=batch_size,
    )
    return (
        model,
        "hf/" + model_name,
        model._langchain_embedding.client.max_seq_length,  # type: ignore
    )


def bench_simple_vector_store(
    embed_models: List[Callable[[int], Tuple[BaseEmbedding, str, int]]],
    num_strings: List[int] = [100],
    string_lengths: List[int] = [128, 512],
    embed_batch_sizes: List[int] = [1, DEFAULT_EMBED_BATCH_SIZE],
    torch_num_threads: Optional[int] = None,
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
                models = []
                for create_model in embed_models:
                    models.append(create_model(batch_size))

                for model in models:
                    for i, string in enumerate(strings):
                        model[0].queue_text_for_embedding(str(i), string)

                    time1 = time.time()
                    _ = model[0].get_queued_text_embeddings(show_progress=True)

                    time2 = time.time()
                    print(
                        f"""Embedding with model {model[1]} with \
batch size {batch_size} and max_seq_length {model[2]} for \
{string_count} strings of length {string_length} took {time2 - time1} seconds"""
                    )
                # TODO: async version


if __name__ == "__main__":
    bench_simple_vector_store(
        embed_models=[
            # create_open_ai_embedding,
            partial(
                create_hf_embedding,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            ),
            partial(
                create_hf_embedding,
                model_name="sentence-transformers/all-mpnet-base-v2",
            ),
        ],
        torch_num_threads=12,
    )
