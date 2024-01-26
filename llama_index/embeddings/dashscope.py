"""DashScope embeddings file."""

import logging
from enum import Enum
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union

from pydantic import PrivateAttr

from llama_index.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)


class DashScopeTextEmbeddingType(str, Enum):
    """DashScope TextEmbedding text_type."""

    TEXT_TYPE_QUERY = "query"
    TEXT_TYPE_DOCUMENT = "document"


class DashScopeTextEmbeddingModels(str, Enum):
    """DashScope TextEmbedding models."""

    TEXT_EMBEDDING_V1 = "text-embedding-v1"
    TEXT_EMBEDDING_V2 = "text-embedding-v2"


class DashScopeBatchTextEmbeddingModels(str, Enum):
    """DashScope TextEmbedding models."""

    TEXT_EMBEDDING_ASYNC_V1 = "text-embedding-async-v1"
    TEXT_EMBEDDING_ASYNC_V2 = "text-embedding-async-v2"


class DashScopeMultiModalEmbeddingModels(str, Enum):
    """DashScope MultiModalEmbedding models."""

    MULTIMODAL_EMBEDDING_ONE_PEACE_V1 = "multimodal-embedding-one-peace-v1"


EMBED_MAX_INPUT_LENGTH = 2048
EMBED_MAX_BATCH_SIZE = 25


def get_text_embedding(
    model: str,
    text: Union[str, List[str]],
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> List[List[float]]:
    """Call DashScope text embedding.
       ref: https://help.aliyun.com/zh/dashscope/developer-reference/text-embedding-api-details.

    Args:
        model (str): The `DashScopeTextEmbeddingModels`
        text (Union[str, List[str]]): text or list text to embedding.

    Raises:
        ImportError: need import dashscope

    Returns:
        List[List[float]]: The list of embedding result, if failed return empty list.
    """
    try:
        import dashscope
    except ImportError:
        raise ImportError("DashScope requires `pip install dashscope")
    if isinstance(text, str):
        text = [text]
    embedding_results = []
    response = dashscope.TextEmbedding.call(
        model=model, input=text, api_key=api_key, kwargs=kwargs
    )
    if response.status_code == HTTPStatus.OK:
        for emb in response.output["embeddings"]:
            embedding_results.append(emb["embedding"])
    else:
        logger.error("Calling TextEmbedding failed, details: %s" % response)

    return embedding_results


def get_batch_text_embedding(
    model: str, url: str, api_key: Optional[str] = None, **kwargs: Any
) -> Optional[str]:
    """Call DashScope batch text embedding.

    Args:
        model (str): The `DashScopeMultiModalEmbeddingModels`
        url (str): The url of the file to embedding which with lines of text to embedding.

    Raises:
        ImportError: Need install dashscope package.

    Returns:
        str: The url of the embedding result, format ref:
        https://help.aliyun.com/zh/dashscope/developer-reference/text-embedding-async-api-details
    """
    try:
        import dashscope
    except ImportError:
        raise ImportError("DashScope requires `pip install dashscope")
    response = dashscope.BatchTextEmbedding.call(
        model=model, url=url, api_key=api_key, kwargs=kwargs
    )
    if response.status_code == HTTPStatus.OK:
        return response.output["url"]
    else:
        logger.error("Calling BatchTextEmbedding failed, details: %s" % response)
        return None


def get_multimodal_embedding(
    model: str, input: list, api_key: Optional[str] = None, **kwargs: Any
) -> List[float]:
    """Call DashScope multimodal embedding.
       ref: https://help.aliyun.com/zh/dashscope/developer-reference/one-peace-multimodal-embedding-api-details.

    Args:
        model (str): The `DashScopeBatchTextEmbeddingModels`
        input (str): The input of the embedding, eg:
             [{'factor': 1, 'text': '你好'},
             {'factor': 2, 'audio': 'https://dashscope.oss-cn-beijing.aliyuncs.com/audios/cow.flac'},
             {'factor': 3, 'image': 'https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png'}]

    Raises:
        ImportError: Need install dashscope package.

    Returns:
        str: The url of the embedding result, format ref:
        https://help.aliyun.com/zh/dashscope/developer-reference/text-embedding-async-api-details
    """
    try:
        import dashscope
    except ImportError:
        raise ImportError("DashScope requires `pip install dashscope")
    response = dashscope.MultiModalEmbedding.call(
        model=model, input=input, api_key=api_key, kwargs=kwargs
    )
    if response.status_code == HTTPStatus.OK:
        return response.output["embedding"]
    else:
        logger.error("Calling MultiModalEmbedding failed, details: %s" % response)
        return []


class DashScopeTextEmbedding(BaseEmbedding):
    """DashScope class for text embedding.

    Args:
        model_name (str): Model name for embedding.
            Defaults to DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2.
                Options are:

                - DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1
                - DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2
        api_key (str): The DashScope api key.
    """

    _api_key: Optional[str] = PrivateAttr()

    def __init__(
        self,
        model_name: str = DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._api_key = api_key
        super().__init__(
            model_name=model_name,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "DashScopeTextEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        emb = get_text_embedding(
            self.model_name,
            query,
            api_key=self._api_key,
            text_type=DashScopeTextEmbeddingType.TEXT_TYPE_QUERY,
        )
        if len(emb) > 0:
            return emb[0]
        else:
            return []

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        emb = get_text_embedding(
            self.model_name,
            text,
            api_key=self._api_key,
            text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
        )
        if len(emb) > 0:
            return emb[0]
        else:
            return []

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return get_text_embedding(
            self.model_name,
            texts,
            api_key=self._api_key,
            text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
        )

    # TODO: use proper async methods
    async def _aget_text_embedding(self, query: str) -> List[float]:
        """Get text embedding."""
        return self._get_text_embedding(query)

    # TODO: user proper async methods
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_query_embedding(query)


class DashScopeBatchTextEmbedding:
    """DashScope class for batch text embedding.You can input a large amount of text to be
       embedding into a file line by line, upload it to the network (it needs to be
       publicly accessible), and call the embedding service through the file URL.
       After the task is completed, the embedding result is saved on the network
       and the result file URL is returned to you.

    Args:
        model_name (str): Model name for embedding.
            Defaults to DashScopeBatchTextEmbeddingModels.TEXT_EMBEDDING_ASYNC_V2.
                Options are:

                - DashScopeBatchTextEmbeddingModels.TEXT_EMBEDDING_ASYNC_V1
                - DashScopeBatchTextEmbeddingModels.TEXT_EMBEDDING_ASYNC_V2
        api_key (str): The DashScope api key.
    """

    _api_key: Optional[str] = PrivateAttr()
    _model_name: str = PrivateAttr()

    def __init__(
        self,
        model_name: str = DashScopeBatchTextEmbeddingModels.TEXT_EMBEDDING_ASYNC_V2,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._api_key = api_key
        self._model_name = model_name

    @classmethod
    def class_name(cls) -> str:
        return "DashScopeBatchTextEmbedding"

    def get_batch_query_embedding(self, embedding_file_url: str) -> Optional[str]:
        """Get batch query embeddings.

        Args:
            embedding_file_url (str): The url of the file to embedding which with lines of text to embedding.

        Returns:
            str: The url of the embedding result, format ref:
                 https://help.aliyun.com/zh/dashscope/developer-reference/text-embedding-async-api-details.
        """
        return get_batch_text_embedding(
            self._model_name,
            embedding_file_url,
            api_key=self._api_key,
            text_type=DashScopeTextEmbeddingType.TEXT_TYPE_QUERY,
        )

    def get_batch_text_embedding(self, embedding_file_url: str) -> Optional[str]:
        """Get batch text embeddings.

        Args:
            embedding_file_url (str): The url of the file to embedding which with lines of text to embedding.

        Returns:
            str: The url of the embedding result, format ref:
                 https://help.aliyun.com/zh/dashscope/developer-reference/text-embedding-async-api-details.
        """
        return get_batch_text_embedding(
            self._model_name,
            embedding_file_url,
            api_key=self._api_key,
            text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
        )


class DashScopeMultiModalEmbedding:
    """DashScope class for multimodal embedding. Images, audio, and text can be input for embedding.

    Args:
        model_name (str): Model name for embedding.
            Defaults to DashScopeMultiModalEmbeddingModels.MULTIMODAL_EMBEDDING_ONE_PEACE_V1.
                Options are:

                - DashScopeMultiModalEmbeddingModels.MULTIMODAL_EMBEDDING_ONE_PEACE_V1.
        api_key (str): The DashScope api key.
        auto_truncation (bool): auto truncation text.
            Defaults to False.
    """

    _api_key: Optional[str] = PrivateAttr()
    _model_name: str = PrivateAttr()

    def __init__(
        self,
        model_name: str = DashScopeMultiModalEmbeddingModels.MULTIMODAL_EMBEDDING_ONE_PEACE_V1,
        api_key: Optional[str] = None,
        auto_truncation: bool = False,
        **kwargs: Any,
    ) -> None:
        self.auto_truncation = auto_truncation
        self._api_key = api_key
        self._model_name = model_name

    @classmethod
    def class_name(cls) -> str:
        return "DashScopeMultiModalEmbedding"

    def get_embedding(self, input: List[Dict]) -> List[float]:
        return get_multimodal_embedding(
            self._model_name,
            input=input,
            api_key=self._api_key,
            auto_truncation=self.auto_truncation,
        )
