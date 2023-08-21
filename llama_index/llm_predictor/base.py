"""Wrapper functions around an LLM chain."""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from pydantic import BaseModel, PrivateAttr

from llama_index.callbacks.base import CallbackManager
from llama_index.llm_predictor.utils import (
    astream_chat_response_to_tokens,
    astream_completion_response_to_tokens,
    stream_chat_response_to_tokens,
    stream_completion_response_to_tokens,
)
from llama_index.llms.base import LLM, ChatMessage, LLMMetadata, MessageRole
from llama_index.llms.generic_utils import messages_to_prompt
from llama_index.llms.utils import LLMType, resolve_llm
from llama_index.prompts.base import (
    BasePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
    SelectorPromptTemplate,
)
from llama_index.types import TokenAsyncGen, TokenGen

logger = logging.getLogger(__name__)


class BaseLLMPredictor(BaseModel, ABC):
    """Base LLM Predictor."""

    @property
    @abstractmethod
    def llm(self) -> LLM:
        """Get LLM."""

    @property
    @abstractmethod
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""

    @abstractmethod
    def predict(self, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
        """Predict the answer to a query."""

    @abstractmethod
    def stream(self, prompt: BasePromptTemplate, **prompt_args: Any) -> TokenGen:
        """Stream the answer to a query."""

    @abstractmethod
    async def apredict(self, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
        """Async predict the answer to a query."""

    @abstractmethod
    async def astream(
        self, prompt: BasePromptTemplate, **prompt_args: Any
    ) -> TokenAsyncGen:
        """Async predict the answer to a query."""


class LLMPredictor(BaseLLMPredictor):
    """LLM predictor class.

    A lightweight wrapper on top of LLMs that handles:
    - conversion of prompts to the string input format expected by LLMs
    - logging of prompts and responses to a callback manager

    NOTE: Mostly keeping around for legacy reasons. A potential future path is to
    deprecate this class and move all functionality into the LLM class.
    """

    class Config:
        arbitrary_types_allowed = True

    system_prompt: Optional[str]
    query_wrapper_prompt: Optional[BasePromptTemplate]
    _llm: LLM = PrivateAttr()

    def __init__(
        self,
        llm: Optional[LLMType] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        query_wrapper_prompt: Optional[BasePromptTemplate] = None,
    ) -> None:
        """Initialize params."""
        self._llm = resolve_llm(llm)

        if callback_manager:
            self._llm.callback_manager = callback_manager

        super().__init__(
            system_prompt=system_prompt, query_wrapper_prompt=query_wrapper_prompt
        )

    @property
    def llm(self) -> LLM:
        """Get LLM."""
        return self._llm

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return self._llm.metadata

    def predict(self, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
        """Predict."""
        if self._llm.metadata.is_chat_model:
            messages = prompt.format_messages(llm=self._llm, **prompt_args)
            messages = self._extend_messages(messages)
            chat_response = self._llm.chat(messages)
            output = chat_response.message.content or ""
            # NOTE: this is an approximation, only for token counting
            formatted_prompt = messages_to_prompt(messages)
        else:
            prompt = self._extend_prompt(prompt)
            formatted_prompt = prompt.format(llm=self._llm, **prompt_args)
            response = self._llm.complete(formatted_prompt)
            output = response.text

        logger.debug(output)

        return output

    def stream(self, prompt: BasePromptTemplate, **prompt_args: Any) -> TokenGen:
        """Stream."""
        if self._llm.metadata.is_chat_model:
            messages = prompt.format_messages(llm=self._llm, **prompt_args)
            messages = self._extend_messages(messages)
            chat_response = self._llm.stream_chat(messages)
            stream_tokens = stream_chat_response_to_tokens(chat_response)
        else:
            prompt = self._extend_prompt(prompt)
            formatted_prompt = prompt.format(llm=self._llm, **prompt_args)
            stream_response = self._llm.stream_complete(formatted_prompt)
            stream_tokens = stream_completion_response_to_tokens(stream_response)
        return stream_tokens

    async def apredict(self, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
        """Async predict."""
        if self._llm.metadata.is_chat_model:
            messages = prompt.format_messages(llm=self._llm, **prompt_args)
            messages = self._extend_messages(messages)
            chat_response = await self._llm.achat(messages)
            output = chat_response.message.content or ""
            # NOTE: this is an approximation, only for token counting
            formatted_prompt = messages_to_prompt(messages)
        else:
            prompt = self._extend_prompt(prompt)
            formatted_prompt = prompt.format(llm=self._llm, **prompt_args)
            response = await self._llm.acomplete(formatted_prompt)
            output = response.text

        logger.debug(output)

        return output

    async def astream(
        self, prompt: BasePromptTemplate, **prompt_args: Any
    ) -> TokenAsyncGen:
        """Async stream."""
        if self._llm.metadata.is_chat_model:
            messages = prompt.format_messages(llm=self._llm, **prompt_args)
            messages = self._extend_messages(messages)
            chat_response = await self._llm.astream_chat(messages)
            stream_tokens = await astream_chat_response_to_tokens(chat_response)
        else:
            prompt = self._extend_prompt(prompt)
            formatted_prompt = prompt.format(llm=self._llm, **prompt_args)
            stream_response = await self._llm.astream_complete(formatted_prompt)
            stream_tokens = await astream_completion_response_to_tokens(stream_response)
        return stream_tokens

    def _extend_prompt(self, prompt: BasePromptTemplate) -> BasePromptTemplate:
        """Add system and query wrapper prompts to base prompt"""
        # TODO: avoid mutating prompt attributes
        if self.system_prompt:
            if isinstance(prompt, SelectorPromptTemplate):
                default_prompt = prompt.default_prompt
                if isinstance(default_prompt, PromptTemplate):
                    default_prompt._template = (
                        self.system_prompt + "\n\n" + default_prompt._template
                    )
                else:
                    raise ValueError("PromptTemplate expected as default_prompt")
            elif isinstance(prompt, ChatPromptTemplate):
                prompt._message_templates = [
                    ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt)
                ] + prompt._message_templates
            elif isinstance(prompt, PromptTemplate):
                prompt._template = self.system_prompt + "\n\n" + prompt._template

        if self.query_wrapper_prompt:
            if isinstance(prompt, (PromptTemplate, ChatPromptTemplate)):
                prompt.kwargs["query_str"] = self.query_wrapper_prompt.format(
                    query_str=prompt.kwargs["query_str"]
                )
            elif isinstance(prompt, SelectorPromptTemplate):
                if isinstance(default_prompt, PromptTemplate):
                    prompt.default_prompt.kwargs[
                        "query_str"
                    ] = self.query_wrapper_prompt.format(
                        query_str=prompt.default_prompt.kwargs["query_str"]
                    )
                else:
                    raise ValueError("PromptTemplate expected as default_prompt")

        return prompt

    def _extend_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Add system prompt to chat message list"""
        if self.system_prompt:
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt)
            ] + messages
        return messages
