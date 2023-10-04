"""Embedding utils for LlamaIndex."""
import os
from typing import List, Optional, Union

from llama_index.bridge.langchain import Embeddings as LCEmbeddings
from llama_index.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface_utils import (
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
    INSTRUCTOR_MODELS,
)
from llama_index.embeddings.instructor import InstructorEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.token_counter.mock_embed_model import MockEmbedding
from llama_index.utils import get_cache_dir

EmbedType = Union[BaseEmbedding, LCEmbeddings, str]


def save_embedding(embedding: List[float], file_path: str) -> None:
    """Save embedding to file."""
    with open(file_path, "w") as f:
        f.write(",".join([str(x) for x in embedding]))


def load_embedding(file_path: str) -> List[float]:
    """Load embedding from file. Will only return first embedding in file."""
    with open(file_path) as f:
        for line in f:
            embedding = [float(x) for x in line.strip().split(",")]
            break
        return embedding


def resolve_embed_model(embed_model: Optional[EmbedType] = None) -> BaseEmbedding:
    """Resolve embed model."""
    if embed_model == "default":
        try:
            embed_model = OpenAIEmbedding()
        except ValueError as e:
            embed_model = "local"
            print(
                "******\n"
                "Could not load OpenAIEmbedding. Using HuggingFaceBgeEmbeddings "
                f"with model_name={DEFAULT_HUGGINGFACE_EMBEDDING_MODEL}. "
                "If you intended to use OpenAI, please check your OPENAI_API_KEY.\n"
                "Original error:\n"
                f"{e!s}"
                "\n******"
            )

    if isinstance(embed_model, str):
        splits = embed_model.split(":", 1)
        is_local = splits[0]
        model_name = splits[1] if len(splits) > 1 else None
        if is_local != "local":
            raise ValueError(
                "embed_model must start with str 'local' or of type BaseEmbedding"
            )

        cache_folder = os.path.join(get_cache_dir(), "models")
        os.makedirs(cache_folder, exist_ok=True)

        if model_name in INSTRUCTOR_MODELS:
            embed_model = InstructorEmbedding(
                model_name=model_name, cache_folder=cache_folder
            )
        else:
            embed_model = HuggingFaceEmbedding(
                model_name=model_name, cache_folder=cache_folder
            )

    if isinstance(embed_model, LCEmbeddings):
        embed_model = LangchainEmbedding(embed_model)

    if embed_model is None:
        print("Embeddings have been explicitly disabled. Using MockEmbedding.")
        embed_model = MockEmbedding(embed_dim=1)

    return embed_model
