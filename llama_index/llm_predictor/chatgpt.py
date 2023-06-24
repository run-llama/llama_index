"""Wrapper functions around an LLM chain."""

import logging
from typing import Any, List, Optional, Union

import openai
from llama_index.bridge.langchain import (
    LLMChain,
    ChatOpenAI,
    BaseMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    BaseLanguageModel,
    BaseMessage,
    PromptTemplate,
    BasePromptTemplate,
)

from llama_index.llm_predictor.base import LLMPredictor
from llama_index.prompts.base import Prompt
from llama_index.utils import ErrorToRetry, retry_on_exceptions_with_backoff

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
        **kwargs: Any
    ) -> None:
        """Initialize params."""
        super().__init__(
            llm=llm or ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"), **kwargs
        )
        self.prepend_messages = prepend_messages

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
