from typing import Any, List, Optional

from llama_index.core.base.embeddings.base_sparse import (
    BaseSparseEmbedding,
    SparseEmbedding,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr

from fastembed import SparseTextEmbedding
from fastembed.sparse.sparse_embedding_base import (
    SparseEmbedding as FastEmbedSparseEmbedding,
)


class FastEmbedSparseEmbedding(BaseSparseEmbedding):
    """
    Qdrant FastEmbedding Sparse models.
    FastEmbed is a lightweight, fast, Python library built for embedding generation.
    See more documentation at:
    * https://github.com/qdrant/fastembed/
    * https://qdrant.github.io/fastembed/.

    To use this class, you must install the `fastembed` Python package.

    `pip install fastembed`
    Example:
        from llama_index.sparse_embeddings.fastembed import FastEmbedSparseEmbedding
        fastembed = FastEmbedSparseEmbedding()
    """

    model_name: str = Field(
        "prithivida/Splade_PP_en_v1",
        description="Name of the FastEmbedding sparse model to use.\n"
        "Defaults to 'prithivida/Splade_PP_en_v1'.\n"
        "Find the list of supported models at "
        "https://qdrant.github.io/fastembed/examples/Supported_Models/",
    )

    max_length: int = Field(
        512,
        description="The maximum number of tokens. Defaults to 512.\n"
        "Unknown behavior for values > 512.",
    )

    cache_dir: Optional[str] = Field(
        None,
        description="The path to the cache directory.\n"
        "Defaults to `local_cache` in the parent directory",
    )

    threads: Optional[int] = Field(
        None,
        description="The number of threads single onnxruntime session can use.\n"
        "Defaults to None",
    )

    _model: SparseTextEmbedding = PrivateAttr()

    @classmethod
    def class_name(self) -> str:
        return "FastEmbedSparseEmbedding"

    def __init__(
        self,
        model_name: Optional[str] = "prithivida/Splade_PP_en_v1",
        max_length: Optional[int] = 512,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[List[Any]] = None,
    ):
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            cache_dir=cache_dir,
            threads=threads,
        )

        self._model = SparseTextEmbedding(
            model_name=model_name,
            max_length=max_length,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
        )

    def _fastembed_to_dict(
        self, fastembed_results: List[FastEmbedSparseEmbedding]
    ) -> List[SparseEmbedding]:
        """Convert FastEmbedSparseEmbedding to SparseEmbedding dict."""
        results = []

        for embedding in fastembed_results:
            result_dict = {}
            for indice, value in zip(embedding.indices, embedding.values):
                result_dict[int(indice)] = float(value)
            results.append(result_dict)

        return results

    def _get_text_embedding(self, text: str) -> SparseEmbedding:
        results = self._model.passage_embed([text])
        return self._fastembed_to_dict(results)[0]

    async def _aget_text_embedding(self, text: str) -> SparseEmbedding:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[SparseEmbedding]:
        results = self._model.passage_embed(texts)
        return self._fastembed_to_dict(results)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[SparseEmbedding]:
        return self._get_text_embeddings(texts)

    def _get_query_embedding(self, query: str) -> SparseEmbedding:
        results = self._model.query_embed(query)
        return self._fastembed_to_dict(results)[0]

    async def _aget_query_embedding(self, query: str) -> SparseEmbedding:
        return self._get_query_embedding(query)
