import time
from typing import List, Optional

from llama_index import SimpleDirectoryReader
from llama_index.embeddings import OpenAIEmbedding
from llama_index.embeddings.base import DEFAULT_EMBED_BATCH_SIZE
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding


def generate_strings(num_strings: int = 100, string_length: int = 10) -> List[str]:
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


def get_max_seq_length(model):  # type: ignore
    return model._langchain_embedding.client.max_seq_length  # type: ignore


def bench_simple_vector_store(
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
                    (
                        "OpenAIEmbedding",
                        4096,
                    ),
                    (
                        "hf/sentence-transformers/all-mpnet-base-v2",
                        get_max_seq_length(embed_models[1]),
                    ),
                    (
                        "hf/sentence-transformers/all-MiniLM-L6-v2",
                        get_max_seq_length(embed_models[2]),
                    ),
                ]

                skip_set = [0]  # skip openai

                for idx, model in enumerate(embed_models):
                    if idx in skip_set:
                        continue
                    for i, string in enumerate(strings):
                        model.queue_text_for_embedding(str(i), string)

                    time1 = time.time()
                    _ = model.get_queued_text_embeddings(show_progress=True)

                    time2 = time.time()
                    print(
                        f"""Embedding with model {embed_model_info[idx][0]} with \
batch size {batch_size} and max_seq_length {embed_model_info[idx][1]} for \
{string_count} strings of length {string_length} took {time2 - time1} seconds"""
                    )
                # TODO: async version


if __name__ == "__main__":
    bench_simple_vector_store(torch_num_threads=12)
