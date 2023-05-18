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
from llama_index.token_counter.token_counter import llm_token_counter
from llama_index.types import RESPONSE_TEXT_TYPE

logger = logging.getLogger(__name__)


class BaseResponseBuilder(ABC):
    """Response builder class."""

    def __init__(
        self,
        service_context: ServiceContext,
        streaming: bool = False,
    ) -> None:
        """Init params."""
        self._service_context = service_context
        self._streaming = streaming

    @property
    def service_context(self) -> ServiceContext:
        return self._service_context

    def _log_prompt_and_response(
        self,
        formatted_prompt: str,
        response: RESPONSE_TEXT_TYPE,
        log_prefix: str = "",
    ) -> None:
        """Log prompt and response from LLM."""
        logger.debug(f"> {log_prefix} prompt template: {formatted_prompt}")
        self._service_context.llama_logger.add_log(
            {"formatted_prompt_template": formatted_prompt}
        )
        logger.debug(f"> {log_prefix} response: {response}")
        self._service_context.llama_logger.add_log(
            {f"{log_prefix.lower()}_response": response or "Empty Response"}
        )

    @abstractmethod
    @llm_token_counter("get_response")
    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response."""
        ...

    @abstractmethod
    @llm_token_counter("aget_response")
    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response."""
        ...
