from lancedb.embeddings import get_registry, EmbeddingFunction

from llama_index.core.bridge.pydantic import BaseModel, Field, model_validator
from typing import Literal, Union, Any
from typing_extensions import Self


class LanceDBTextModel(BaseModel):
    embedding_model: Union[
        Literal[
            "bedrock-text",
            "cohere",
            "gemini-text",
            "instructor",
            "ollama",
            "openai",
            "sentence-transformers",
            "gte-text",
            "huggingface",
            "colbert",
            "jina",
            "watsonx",
            "voyageai",
        ],
        EmbeddingFunction,
    ]
    kwargs: dict = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_embedding_model(self) -> Self:
        if isinstance(self.embedding_model, str):
            try:
                self.embedding_model = (
                    get_registry().get(self.embedding_model).create(**self.kwargs)
                )
            except Exception as e:
                return ValueError(
                    f"An exception occurred while creating the embeddings function: {e}"
                )
        return self


class LanceDBMultiModalModel(BaseModel):
    embedding_model: Union[
        Literal["open-clip", "colpali", "jina", "imagebind"], EmbeddingFunction
    ]
    kwargs: dict = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_embedding_model(self) -> Self:
        if isinstance(self.embedding_model, str):
            try:
                self.embedding_model = (
                    get_registry().get(self.embedding_model).create(**self.kwargs)
                )
            except Exception as e:
                return ValueError(
                    f"An exception occurred while creating the embeddings function: {e}"
                )
        return self


def get_lancedb_text_embedding_model(
    embedding_model: Literal[
        "bedrock-text",
        "cohere",
        "gemini-text",
        "instructor",
        "ollama",
        "openai",
        "sentence-transformers",
        "gte-text",
        "huggingface",
        "colbert",
        "jina",
        "watsonx",
        "voyageai",
    ],
    **kwargs: Any,
):
    """
    Get a pre-defined LanceDB text embedding model.

    Args:
        embedding_model (str): name of the embedding model.
        **kwargs (Any): keyword arguments that are necessary for the initialization of the embedding model you want to use.

    Returns:
        EmbeddingFunction: a LanceDB embedding function.

    """
    return LanceDBTextModel(embedding_model=embedding_model, kwargs=kwargs)


def get_lancedb_multimodal_embedding_model(
    embedding_model: Literal["open-clip", "colpali", "jina", "imagebind"], **kwargs: Any
):
    """
    Get a pre-defined LanceDB multimodal embedding model.

    Args:
        embedding_model (str): name of the embedding model.
        **kwargs (Any): keyword arguments that are necessary for the initialization of the embedding model you want to use.

    Returns:
        EmbeddingFunction: a LanceDB embedding function.

    """
    return LanceDBMultiModalModel(embedding_model=embedding_model, kwargs=kwargs)
