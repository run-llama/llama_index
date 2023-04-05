"""Wrapper functions around an LLM chain."""

import logging
from typing import Any, List, Optional, Union, Tuple, cast

import openai
from langchain import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import ChatOpenAI
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    BaseMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseLanguageModel, BaseMessage
from langchain.callbacks.base import BaseCallbackManager

from gpt_index.llm_predictor.base import LLMPredictor
from gpt_index.prompts.base import Prompt
from gpt_index.utils import ErrorToRetry, retry_on_exceptions_with_backoff

logger = logging.getLogger(__name__)


class ChatGPTLLMPredictor(LLMPredictor):
    """ChatGPT Specific LLM predictor class.

    Wrapper around an LLMPredictor to provide ChatGPT specific features.

    Args:
        llm (Optional[langchain.llms.base.LLM]): LLM from Langchain to use
            for predictions. Defaults to OpenAI's text-davinci-003 model.
            Please see `Langchain's LLM Page
            <https://langchain.readthedocs.io/en/latest/modules/llms.html>`_
            for more details.

        retry_on_throttling (bool): Whether to retry on rate limit errors.
            Defaults to true.

    """

    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        prepend_messages: Optional[
            List[Union[BaseMessagePromptTemplate, BaseMessage]]
        ] = None,
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> None:
        """Initialize params."""
        super().__init__(
            llm=llm or ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"), **kwargs
        )
        self.prepend_messages = prepend_messages
        self._callback_manager = callback_manager

    def _get_langchain_prompt(
        self, prompt: Prompt
    ) -> Union[ChatPromptTemplate, BasePromptTemplate]:
        """Add prepend_messages to prompt."""
        lc_prompt = prompt.get_langchain_prompt(llm=self._llm)
        if self.prepend_messages:
            if isinstance(lc_prompt, PromptTemplate):
                msgs = self.prepend_messages + [
                    HumanMessagePromptTemplate.from_template(lc_prompt.template)
                ]
                lc_prompt = ChatPromptTemplate.from_messages(msgs)
            elif isinstance(lc_prompt, ChatPromptTemplate):
                lc_prompt.messages = self.prepend_messages + lc_prompt.messages

        return lc_prompt

    def _predict(self, prompt: Prompt, **prompt_args: Any) -> str:
        """Inner predict function.

        If retry_on_throttling is true, we will retry on rate limit errors.

        """
        lc_prompt = self._get_langchain_prompt(prompt)
        llm_chain = LLMChain(prompt=lc_prompt, llm=self._llm)

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

    async def _apredict(self, prompt: Prompt, **prompt_args: Any) -> str:
        """Async inner predict function.

        If retry_on_throttling is true, we will retry on rate limit errors.

        """
        lc_prompt = self._get_langchain_prompt(prompt)
        llm_chain = LLMChain(prompt=lc_prompt, llm=self._llm)

        # Note: we don't pass formatted_prompt to llm_chain.predict because
        # langchain does the same formatting under the hood
        full_prompt_args = prompt.get_full_format_args(prompt_args)
        # TODO: support retry on throttling
        llm_prediction = await llm_chain.apredict(**full_prompt_args)
        return llm_prediction

    def predict_with_stream(
        self, prompt: Prompt, is_last: bool = False, **prompt_args: Any
    ) -> Tuple[str, str]:
        """Predict the answer to a query.

        This mode will also call the callback manager, only if it's the final response.
        Specifically, it will call `on_llm_new_token`.

        If it's an intermediate response, doesn't make sense to stream.

        """
        # try casting to validate
        try:
            llm = cast(BaseChatModel, self._llm)
        except Exception:
            raise ValueError("LLM must be a BaseChatModel to use predict_with_stream")

        if self._callback_manager is None:
            raise ValueError(
                "Callback manager must be set to use predict_with_stream mode"
            )

        # only set callback manager if it's the last response generated from the LLM
        if not is_last:
            llm.callback_manager = None
        else:
            llm.callback_manager = self._callback_manager

        return self.predict(prompt, **prompt_args)
