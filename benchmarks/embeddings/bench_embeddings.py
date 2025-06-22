import time
from functools import partial
from typing import Callable, List, Optional, Tuple

import pandas as pd
from llama_index.core import SimpleDirectoryReader
from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)


def generate_strings(num_strings: int = 100, string_length: int = 10) -> List[str]:
    """
    Generate random strings sliced from the paul graham essay.

    Has the following form:

    offset 0: [0:string_length], [string_length:2*string_length], ...
    offset 1: [1:1+string_length], [1+string_length:1+2*string_length],...
    ...
    """  # noqa: D415
    content = (
        SimpleDirectoryReader("../../docs/docs/examples/data/paul_graham")
        .load_data()[0]
        .get_content()
    )
    content_length = len(content)

    strings_per_loop = content_length / string_length
    num_loops_upper_bound = int(num_strings / strings_per_loop) + 1
    strings = []

    for offset in range(num_loops_upper_bound + 1):
        ptr = offset % string_length
        while ptr + string_length < content_length:
            strings.append(content[ptr : ptr + string_length])
            ptr += string_length
            if len(strings) == num_strings:
                break

    return strings


def create_open_ai_embedding(batch_size: int) -> Tuple[BaseEmbedding, str, int]:
    from llama_index.embeddings.openai import OpenAIEmbedding

    return (
        OpenAIEmbedding(embed_batch_size=batch_size),
        "OpenAIEmbedding",
        4096,
    )


def create_local_embedding(
    model_name: str, batch_size: int, **kwargs: Optional[dict]
) -> Tuple[BaseEmbedding, str, int]:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    model = HuggingFaceEmbedding(model_name, embed_batch_size=batch_size, **kwargs)
    return (
        model,
        "hf/" + model_name,
        model._model.max_seq_length,  # type: ignore
    )


def bench_simple_vector_store(
    embed_models: List[Callable[[int], Tuple[BaseEmbedding, str, int]]],
    num_strings: List[int] = [100],
    string_lengths: List[int] = [64, 256],
    embed_batch_sizes: List[int] = [
        DEFAULT_EMBED_BATCH_SIZE,
        2 * DEFAULT_EMBED_BATCH_SIZE,
    ],
    torch_num_threads: Optional[int] = None,
) -> None:
    """Benchmark embeddings."""
    print("Benchmarking Embeddings\n---------------------------")

    results = []

    if torch_num_threads is not None:
        import torch  # pants: no-infer-dep

        torch.set_num_threads(torch_num_threads)

    max_num_strings = max(num_strings)
    for string_length in string_lengths:
        generated_strings = generate_strings(
            num_strings=max_num_strings, string_length=string_length
        )

        for string_count in num_strings:
            strings = generated_strings[:string_count]

            for batch_size in embed_batch_sizes:
                models = []
                for create_model in embed_models:
                    models.append(create_model(batch_size=batch_size))  # type: ignore

                for model in models:
                    time1 = time.time()
                    _ = model[0].get_text_embedding_batch(strings, show_progress=True)

                    time2 = time.time()
                    print(
                        f"Embedding with model {model[1]} with "
                        f"batch size {batch_size} and max_seq_length {model[2]} for "
                        f"{string_count} strings of length {string_length} took "
                        f"{time2 - time1} seconds."
                    )
                    results.append(
                        (
                            model[1],
                            batch_size,
                            string_length,
                            batch_size / (time2 - time1),
                        )
                    )
                # TODO: async version

    # print final results
    print("\n\nFinal Results\n---------------------------")
    results_df = pd.DataFrame(
        results, columns=["model", "batch_size", "string_length", "strings_per_second"]
    )
    print(results_df)


if __name__ == "__main__":
    bench_simple_vector_store(
        embed_models=[
            # create_open_ai_embedding,
            partial(
                create_local_embedding,
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # 22.7M params
            ),
            partial(
                create_local_embedding,
                model_name="ibm-granite/granite-embedding-30m-english",  # 30.3M params
            ),
            partial(
                create_local_embedding,
                model_name="sentence-transformers/all-MiniLM-L12-v2",  # 33.4M params
            ),
            partial(
                create_local_embedding,
                model_name="BAAI/bge-small-en-v1.5",  # 33.4M params
            ),
            partial(
                create_local_embedding,
                model_name="sentence-transformers/all-mpnet-base-v2",  # 109M params
            ),
            partial(
                create_local_embedding,
                model_name="ibm-granite/granite-embedding-125m-english",  # 125M params
            ),
            partial(
                create_local_embedding,
                model_name="nomic-ai/nomic-embed-text-v1.5",  # 137M params
                trust_remote_code=True,
            ),
            partial(
                create_local_embedding,
                model_name="Alibaba-NLP/gte-modernbert-base",  # 149M params
            ),
            partial(
                create_local_embedding,
                model_name="mixedbread-ai/mxbai-embed-large-v1",  # 335M params
            ),
            partial(
                create_local_embedding,
                model_name="BAAI/bge-large-en-v1.5",  # 335M params
            ),
            partial(
                create_local_embedding,
                model_name="Snowflake/snowflake-arctic-embed-l-v2.0",  # 568M params
            ),
        ],
        torch_num_threads=None,
    )
