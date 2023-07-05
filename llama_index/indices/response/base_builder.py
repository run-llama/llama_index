"""Response builder class.

This class provides general functions for taking in a set of text
and generating a response.

Will support different modes, from 1) stuffing chunks into prompt,
2) create and refine separately over each chunk, 3) tree summarization.

"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from llama_index.indices.service_context import ServiceContext
from llama_index.types import RESPONSE_TEXT_TYPE

logger = logging.getLogger(__name__)


class BaseResponseBuilder(ABC):
    """Response builder class."""

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        streaming: bool = False,
    ) -> None:
        """Init params."""
        self._service_context = (
            service_context
            or ServiceContext.get_global()
            or ServiceContext.from_defaults()
        )
        self._streaming = streaming

    @property
    def service_context(self) -> ServiceContext:
        return self._service_context

    @abstractmethod
    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response."""
        ...

    @abstractmethod
    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response."""
        ...
