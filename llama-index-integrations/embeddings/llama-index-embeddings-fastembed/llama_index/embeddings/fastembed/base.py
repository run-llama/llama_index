from typing import Any, List, Literal, Optional

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr

from fastembed import TextEmbedding


class FastEmbedEmbedding(BaseEmbedding):
    """
    Qdrant FastEmbedding models.
    FastEmbed is a lightweight, fast, Python library built for embedding generation.
    See more documentation at:
    * https://github.com/qdrant/fastembed/
    * https://qdrant.github.io/fastembed/.

    To use this class, you must install the `fastembed` Python package.

    `pip install fastembed`
    Example:
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
        fastembed = FastEmbedEmbedding()
    """

    model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Name of the FastEmbedding model to use.\n"
        "Find the list of supported models at "
        "https://qdrant.github.io/fastembed/examples/Supported_Models/",
    )

    max_length: int = Field(
        default=512,
        description="The maximum number of tokens. Defaults to 512.\n"
        "Unknown behavior for values > 512.",
    )

    cache_dir: Optional[str] = Field(
        default=None,
        description="The path to the cache directory.\n"
        "Defaults to `local_cache` in the parent directory",
    )

    threads: Optional[int] = Field(
        default=None,
        description="The number of threads single onnxruntime session can use.\n"
        "Defaults to None",
    )

    doc_embed_type: Literal["default", "passage"] = Field(
        default="default",
        description="Type of embedding method to use for documents.\n"
        "Available options are 'default' and 'passage'.",
    )

    providers: Optional[List[str]] = Field(
        default=None,
        description="The ONNX providers to use for the embedding model.",
    )

    _model: TextEmbedding = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        return "FastEmbedEmbedding"

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        max_length: Optional[int] = 512,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        doc_embed_type: Literal["default", "passage"] = "default",
        providers: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            **kwargs,
        )

        self._model = TextEmbedding(
            model_name=model_name,
            max_length=max_length,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            **kwargs,
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[np.ndarray]
        if self.doc_embed_type == "passage":
            embeddings = list(self._model.passage_embed(texts))
        else:
            embeddings = list(self._model.embed(texts))
        return [embedding.tolist() for embedding in embeddings]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)

    def _get_query_embedding(self, query: str) -> List[float]:
        query_embeddings: list[np.ndarray] = list(self._model.query_embed(query))
        return query_embeddings[0].tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
