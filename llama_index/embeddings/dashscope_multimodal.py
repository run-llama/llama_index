"""DashScope multimodal embeddings file."""

import logging
from enum import Enum
from http import HTTPStatus
from typing import Any, Dict, List, Optional

from pydantic import PrivateAttr

logger = logging.getLogger(__name__)


class DashScopeMultiModalEmbeddingModels(str, Enum):
    """DashScope MultiModalEmbedding models."""

    MULTIMODAL_EMBEDDING_ONE_PEACE_V1 = "multimodal-embedding-one-peace-v1"


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
