"""Wrapper functions around an LLM chain."""

import logging
from abc import abstractmethod
from dataclasses import dataclass
from threading import Thread
from typing import Any, Generator, Optional, Protocol, Tuple, runtime_checkable

import openai
from llama_index.bridge.langchain import langchain
from llama_index.bridge.langchain import BaseCache, Cohere, LLMChain, OpenAI
from llama_index.bridge.langchain import ChatOpenAI, AI21, BaseLanguageModel

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.constants import (
    AI21_J2_CONTEXT_WINDOW,
    COHERE_CONTEXT_WINDOW,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.langchain_helpers.streaming import StreamingGeneratorCallbackHandler
from llama_index.llm_predictor.openai_utils import openai_modelname_to_contextsize
from llama_index.prompts.base import Prompt
from llama_index.utils import (
    ErrorToRetry,
    globals_helper,
    retry_on_exceptions_with_backoff,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMMetadata:
    """LLM metadata.

    We extract this metadata to help with our prompts.

    """

    context_window: int = DEFAULT_CONTEXT_WINDOW
    num_output: int = DEFAULT_NUM_OUTPUTS


def _get_llm_metadata(llm: BaseLanguageModel) -> LLMMetadata:
    """Get LLM metadata from llm."""
    if not isinstance(llm, BaseLanguageModel):
        raise ValueError("llm must be an instance of langchain.llms.base.LLM")
    if isinstance(llm, OpenAI):
        return LLMMetadata(
            context_window=openai_modelname_to_contextsize(llm.model_name),
            num_output=llm.max_tokens,
        )
    elif isinstance(llm, ChatOpenAI):
        return LLMMetadata(
            context_window=openai_modelname_to_contextsize(llm.model_name),
            num_output=llm.max_tokens or -1,
        )
    elif isinstance(llm, Cohere):
        # June 2023: Cohere's supported max input size for Generation models is 2048
        # Reference: <https://docs.cohere.com/docs/tokens>
        return LLMMetadata(
            context_window=COHERE_CONTEXT_WINDOW, num_output=llm.max_tokens
        )
    elif isinstance(llm, AI21):
        # June 2023:
        #   AI21's supported max input size for
        #   J2 models is 8K (8192 tokens to be exact)
        # Reference: <https://docs.ai21.com/changelog/increased-context-length-for-j2-foundation-models>  # noqa
        return LLMMetadata(
            context_window=AI21_J2_CONTEXT_WINDOW, num_output=llm.maxTokens
        )
    else:
        return LLMMetadata()


@runtime_checkable
class BaseLLMPredictor(Protocol):
    """Base LLM Predictor."""

    callback_manager: CallbackManager

    @abstractmethod
    def get_llm_metadata(self) -> LLMMetadata:
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

    @property
    @abstractmethod
    def total_tokens_used(self) -> int:
        """Get the total tokens used so far."""

    @property
    @abstractmethod
    def last_token_usage(self) -> int:
        """Get the last token usage."""

    @last_token_usage.setter
    @abstractmethod
    def last_token_usage(self, value: int) -> None:
        """Set the last token usage."""

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

        retry_on_throttling (bool): Whether to retry on rate limit errors.
            Defaults to true.

        cache (Optional[langchain.cache.BaseCache]) : use cached result for LLM
    """

    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        retry_on_throttling: bool = True,
        cache: Optional[BaseCache] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Initialize params."""
        self._llm = llm or OpenAI(
            temperature=0, model_name="text-davinci-003", max_tokens=-1
        )
        if cache is not None:
            langchain.llm_cache = cache
        self.callback_manager = callback_manager or CallbackManager([])
        self.retry_on_throttling = retry_on_throttling
        self._total_tokens_used = 0
        self.flag = True
        self._last_token_usage: Optional[int] = None

    @property
    def llm(self) -> BaseLanguageModel:
        """Get LLM."""
        return self._llm

    def get_llm_metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # TODO: refactor mocks in unit tests, this is a stopgap solution
        if hasattr(self, "_llm") and self._llm is not None:
            return _get_llm_metadata(self._llm)
        else:
            return LLMMetadata()

    def _predict(self, prompt: Prompt, **prompt_args: Any) -> str:
        """Inner predict function.

        If retry_on_throttling is true, we will retry on rate limit errors.

        """
        llm_chain = LLMChain(
            prompt=prompt.get_langchain_prompt(llm=self._llm), llm=self._llm
        )

        # Note: we don't pass formatted_prompt to llm_chain.predict because
        # langchain does the same formatting under the hood
        full_prompt_args = prompt.get_full_format_args(prompt_args)
        if self.retry_on_throttling:
            llm_prediction = retry_on_exceptions_with_backoff(
                lambda: llm_chain.predict(**full_prompt_args),
                [
                    ErrorToRetry(openai.error.RateLimitError),
                    ErrorToRetry(openai.error.ServiceUnavailableError),
                    ErrorToRetry(openai.error.TryAgain),
                    ErrorToRetry(
                        openai.error.APIConnectionError, lambda e: e.should_retry
                    ),
                ],
            )
        else:
            llm_prediction = llm_chain.predict(**full_prompt_args)
        return llm_prediction

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
        llm_prediction = self._predict(prompt, **prompt_args)
        logger.debug(llm_prediction)

        # We assume that the value of formatted_prompt is exactly the thing
        # eventually sent to OpenAI, or whatever LLM downstream
        prompt_tokens_count = self._count_tokens(formatted_prompt)
        prediction_tokens_count = self._count_tokens(llm_prediction)
        self._total_tokens_used += prompt_tokens_count + prediction_tokens_count
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

    @property
    def total_tokens_used(self) -> int:
        """Get the total tokens used so far."""
        return self._total_tokens_used

    def _count_tokens(self, text: str) -> int:
        tokens = globals_helper.tokenizer(text)
        return len(tokens)

    @property
    def last_token_usage(self) -> int:
        """Get the last token usage."""
        if self._last_token_usage is None:
            return 0
        return self._last_token_usage

    @last_token_usage.setter
    def last_token_usage(self, value: int) -> None:
        """Set the last token usage."""
        self._last_token_usage = value

    async def _apredict(self, prompt: Prompt, **prompt_args: Any) -> str:
        """Async inner predict function.

        If retry_on_throttling is true, we will retry on rate limit errors.

        """
        llm_chain = LLMChain(
            prompt=prompt.get_langchain_prompt(llm=self._llm), llm=self._llm
        )

        # Note: we don't pass formatted_prompt to llm_chain.predict because
        # langchain does the same formatting under the hood
        full_prompt_args = prompt.get_full_format_args(prompt_args)
        # TODO: support retry on throttling
        llm_prediction = await llm_chain.apredict(**full_prompt_args)
        return llm_prediction

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
        llm_prediction = await self._apredict(prompt, **prompt_args)
        logger.debug(llm_prediction)

        # We assume that the value of formatted_prompt is exactly the thing
        # eventually sent to OpenAI, or whatever LLM downstream
        prompt_tokens_count = self._count_tokens(formatted_prompt)
        prediction_tokens_count = self._count_tokens(llm_prediction)
        self._total_tokens_used += prompt_tokens_count + prediction_tokens_count
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
