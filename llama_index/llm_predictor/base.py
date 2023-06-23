"""Wrapper functions around an LLM chain."""

import logging
from abc import abstractmethod
from threading import Thread
from typing import Any, Generator, Optional, Protocol, Tuple, runtime_checkable

from langchain import BaseCache

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.langchain_helpers.streaming import StreamingGeneratorCallbackHandler
from llama_index.llms.base import LLMMetadata
from llama_index.llms.utils import LLMType, resolve_llm
from llama_index.prompts.base import Prompt
from llama_index.utils import (
    ErrorToRetry,
    globals_helper,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseLLMPredictor(Protocol):
    """Base LLM Predictor."""

    callback_manager: CallbackManager

    @property
    @abstractmethod
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""

    @abstractmethod
    def predict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Predict the answer to a query.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """

    @abstractmethod
    def stream(self, prompt: Prompt, **prompt_args: Any) -> Tuple[Generator, str]:
        """Stream the answer to a query.

        NOTE: this is a beta feature. Will try to build or use
        better abstractions about response handling.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            str: The predicted answer.

        """

    @abstractmethod
    async def apredict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Async predict the answer to a query.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """


class LLMPredictor(BaseLLMPredictor):
    """LLM predictor class.

    Wrapper around an LLMChain from Langchain.

    Args:
        llm (Optional[langchain.llms.base.LLM]): LLM from Langchain to use
            for predictions. Defaults to OpenAI's text-davinci-003 model.
            Please see `Langchain's LLM Page
            <https://langchain.readthedocs.io/en/latest/modules/llms.html>`_
            for more details.
        cache (Optional[langchain.cache.BaseCache]) : use cached result for LLM
    """

    def __init__(
        self,
        llm: Optional[LLMType] = None,
        cache: Optional[BaseCache] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Initialize params."""
        self._llm = resolve_llm(llm)
        # TODO: move this to be langchain specific
        # if cache is not None:
        #     langchain.llm_cache = cache
        self.callback_manager = callback_manager or CallbackManager([])

    def get_llm_metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return self._llm.metadata

    def predict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Predict the answer to a query.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """
        llm_payload = {**prompt_args}
        llm_payload[EventPayload.TEMPLATE] = prompt
        event_id = self.callback_manager.on_event_start(
            CBEventType.LLM,
            payload=llm_payload,
        )
        formatted_prompt = prompt.format(llm=self._llm, **prompt_args)
        output = self._llm.complete(formatted_prompt)
        llm_prediction = str(output)
        # TODO: normalize this into a string output
        logger.debug(llm_prediction)

        # We assume that the value of formatted_prompt is exactly the thing
        # eventually sent to OpenAI, or whatever LLM downstream
        prompt_tokens_count = self._count_tokens(formatted_prompt)
        prediction_tokens_count = self._count_tokens(llm_prediction)
        self.callback_manager.on_event_end(
            CBEventType.LLM,
            payload={
                EventPayload.RESPONSE: llm_prediction,
                EventPayload.PROMPT: formatted_prompt,
                # deprecated
                "formatted_prompt_tokens_count": prompt_tokens_count,
                "prediction_tokens_count": prediction_tokens_count,
                "total_tokens_used": prompt_tokens_count + prediction_tokens_count,
            },
            event_id=event_id,
        )
        return llm_prediction, formatted_prompt

    def stream(self, prompt: Prompt, **prompt_args: Any) -> Tuple[Generator, str]:
        """Stream the answer to a query.

        NOTE: this is a beta feature. Will try to build or use
        better abstractions about response handling.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            str: The predicted answer.

        """
        formatted_prompt = prompt.format(llm=self._llm, **prompt_args)

        handler = StreamingGeneratorCallbackHandler()

        if not hasattr(self._llm, "callbacks"):
            raise ValueError("LLM must support callbacks to use streaming.")

        self._llm.callbacks = [handler]

        if not getattr(self._llm, "streaming", False):
            raise ValueError("LLM must support streaming and set streaming=True.")

        thread = Thread(target=self._predict, args=[prompt], kwargs=prompt_args)
        thread.start()

        response_gen = handler.get_response_gen()

        # NOTE/TODO: token counting doesn't work with streaming
        return response_gen, formatted_prompt

    def _count_tokens(self, text: str) -> int:
        tokens = globals_helper.tokenizer(text)
        return len(tokens)

    async def apredict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Async predict the answer to a query.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """
        llm_payload = {**prompt_args}
        llm_payload[EventPayload.TEMPLATE] = prompt
        event_id = self.callback_manager.on_event_start(
            CBEventType.LLM, payload=llm_payload
        )
        formatted_prompt = prompt.format(llm=self._llm, **prompt_args)
        llm_prediction = await self._llm.acomplete(formatted_prompt)
        logger.debug(llm_prediction)

        # We assume that the value of formatted_prompt is exactly the thing
        # eventually sent to OpenAI, or whatever LLM downstream
        prompt_tokens_count = self._count_tokens(formatted_prompt)
        prediction_tokens_count = self._count_tokens(llm_prediction)
        self.callback_manager.on_event_end(
            CBEventType.LLM,
            payload={
                EventPayload.RESPONSE: llm_prediction,
                EventPayload.PROMPT: formatted_prompt,
                # deprecated
                "formatted_prompt_tokens_count": prompt_tokens_count,
                "prediction_tokens_count": prediction_tokens_count,
                "total_tokens_used": prompt_tokens_count + prediction_tokens_count,
            },
            event_id=event_id,
        )
        return llm_prediction, formatted_prompt
