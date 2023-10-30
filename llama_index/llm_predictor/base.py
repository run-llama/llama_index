"""Wrapper functions around an LLM chain."""

import logging
from abc import ABC, abstractmethod
from collections import ChainMap
from typing import Any, List, Optional

from llama_index.bridge.pydantic import BaseModel, PrivateAttr
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.llm_predictor.utils import (
    astream_chat_response_to_tokens,
    astream_completion_response_to_tokens,
    stream_chat_response_to_tokens,
    stream_completion_response_to_tokens,
)
from llama_index.llms.base import LLM, ChatMessage, LLMMetadata, MessageRole
from llama_index.llms.utils import LLMType, resolve_llm
from llama_index.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.schema import BaseComponent
from llama_index.types import PydanticProgramMode, TokenAsyncGen, TokenGen

logger = logging.getLogger(__name__)


class BaseLLMPredictor(BaseComponent, ABC):
    """Base LLM Predictor."""

    @property
    @abstractmethod
    def llm(self) -> LLM:
        """Get LLM."""

    @property
    @abstractmethod
    def callback_manager(self) -> CallbackManager:
        """Get callback manager."""

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
    pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT

    _llm: LLM = PrivateAttr()

    def __init__(
        self,
        llm: Optional[LLMType] = "default",
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        query_wrapper_prompt: Optional[BasePromptTemplate] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
    ) -> None:
        """Initialize params."""
        self._llm = resolve_llm(llm)

        if callback_manager:
            self._llm.callback_manager = callback_manager

        super().__init__(
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            pydantic_program_mode=pydantic_program_mode,
        )

    @classmethod
    def class_name(cls) -> str:
        return "LLMPredictor"

    @property
    def llm(self) -> LLM:
        """Get LLM."""
        return self._llm

    @property
    def callback_manager(self) -> CallbackManager:
        """Get callback manager."""
        return self._llm.callback_manager

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return self._llm.metadata

    def _log_template_data(
        self, prompt: BasePromptTemplate, **prompt_args: Any
    ) -> None:
        template_vars = {
            k: v
            for k, v in ChainMap(prompt.kwargs, prompt_args).items()
            if k in prompt.template_vars
        }
        with self.callback_manager.event(
            CBEventType.TEMPLATING,
            payload={
                EventPayload.TEMPLATE: prompt.get_template(llm=self._llm),
                EventPayload.TEMPLATE_VARS: template_vars,
                EventPayload.SYSTEM_PROMPT: self.system_prompt,
                EventPayload.QUERY_WRAPPER_PROMPT: self.query_wrapper_prompt,
            },
        ):
            pass

    def _run_program(
        self,
        output_cls: BaseModel,
        prompt: PromptTemplate,
        **prompt_args: Any,
    ) -> str:
        from llama_index.program.utils import get_program_for_llm

        program = get_program_for_llm(
            output_cls,
            prompt,
            self._llm,
            pydantic_program_mode=self.pydantic_program_mode,
        )

        chat_response = program(**prompt_args)
        return chat_response.json()

    async def _arun_program(
        self,
        output_cls: BaseModel,
        prompt: PromptTemplate,
        **prompt_args: Any,
    ) -> str:
        from llama_index.program.utils import get_program_for_llm

        program = get_program_for_llm(
            output_cls,
            prompt,
            self._llm,
            pydantic_program_mode=self.pydantic_program_mode,
        )

        chat_response = await program.acall(**prompt_args)
        return chat_response.json()

    def predict(
        self,
        prompt: BasePromptTemplate,
        output_cls: Optional[BaseModel] = None,
        **prompt_args: Any,
    ) -> str:
        """Predict."""
        self._log_template_data(prompt, **prompt_args)

        if output_cls is not None:
            output = self._run_program(output_cls, prompt, **prompt_args)
        elif self._llm.metadata.is_chat_model:
            messages = prompt.format_messages(llm=self._llm, **prompt_args)
            messages = self._extend_messages(messages)
            chat_response = self._llm.chat(messages)
            output = chat_response.message.content or ""
        else:
            formatted_prompt = prompt.format(llm=self._llm, **prompt_args)
            formatted_prompt = self._extend_prompt(formatted_prompt)
            response = self._llm.complete(formatted_prompt)
            output = response.text

        logger.debug(output)

        return output

    def stream(
        self,
        prompt: BasePromptTemplate,
        output_cls: Optional[BaseModel] = None,
        **prompt_args: Any,
    ) -> TokenGen:
        """Stream."""
        if output_cls is not None:
            raise NotImplementedError("Streaming with output_cls not supported.")

        self._log_template_data(prompt, **prompt_args)

        if self._llm.metadata.is_chat_model:
            messages = prompt.format_messages(llm=self._llm, **prompt_args)
            messages = self._extend_messages(messages)
            chat_response = self._llm.stream_chat(messages)
            stream_tokens = stream_chat_response_to_tokens(chat_response)
        else:
            formatted_prompt = prompt.format(llm=self._llm, **prompt_args)
            formatted_prompt = self._extend_prompt(formatted_prompt)
            stream_response = self._llm.stream_complete(formatted_prompt)
            stream_tokens = stream_completion_response_to_tokens(stream_response)
        return stream_tokens

    async def apredict(
        self,
        prompt: BasePromptTemplate,
        output_cls: Optional[BaseModel] = None,
        **prompt_args: Any,
    ) -> str:
        """Async predict."""
        self._log_template_data(prompt, **prompt_args)

        if output_cls is not None:
            output = await self._arun_program(output_cls, prompt, **prompt_args)
        elif self._llm.metadata.is_chat_model:
            messages = prompt.format_messages(llm=self._llm, **prompt_args)
            messages = self._extend_messages(messages)
            chat_response = await self._llm.achat(messages)
            output = chat_response.message.content or ""
        else:
            formatted_prompt = prompt.format(llm=self._llm, **prompt_args)
            formatted_prompt = self._extend_prompt(formatted_prompt)
            response = await self._llm.acomplete(formatted_prompt)
            output = response.text

        logger.debug(output)

        return output

    async def astream(
        self,
        prompt: BasePromptTemplate,
        output_cls: Optional[BaseModel] = None,
        **prompt_args: Any,
    ) -> TokenAsyncGen:
        """Async stream."""
        if output_cls is not None:
            raise NotImplementedError("Streaming with output_cls not supported.")

        self._log_template_data(prompt, **prompt_args)

        if self._llm.metadata.is_chat_model:
            messages = prompt.format_messages(llm=self._llm, **prompt_args)
            messages = self._extend_messages(messages)
            chat_response = await self._llm.astream_chat(messages)
            stream_tokens = await astream_chat_response_to_tokens(chat_response)
        else:
            formatted_prompt = prompt.format(llm=self._llm, **prompt_args)
            formatted_prompt = self._extend_prompt(formatted_prompt)
            stream_response = await self._llm.astream_complete(formatted_prompt)
            stream_tokens = await astream_completion_response_to_tokens(stream_response)
        return stream_tokens

    def _extend_prompt(
        self,
        formatted_prompt: str,
    ) -> str:
        """Add system and query wrapper prompts to base prompt."""
        extended_prompt = formatted_prompt
        if self.system_prompt:
            extended_prompt = self.system_prompt + "\n\n" + extended_prompt

        if self.query_wrapper_prompt:
            extended_prompt = self.query_wrapper_prompt.format(
                query_str=extended_prompt
            )

        return extended_prompt

    def _extend_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Add system prompt to chat message list."""
        if self.system_prompt:
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
                *messages,
            ]
        return messages
