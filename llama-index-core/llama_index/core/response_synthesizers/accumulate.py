import asyncio
from typing import Any, Callable, List, Optional, Sequence, Type

from llama_index.core.base.llms.types import ChatMessage

from llama_index.core.async_utils import run_async_tasks
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper, ChatPromptHelper
from llama_index.core.llms import LLM
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.chat_prompts import CHAT_CONTENT_QA_PROMPT
from llama_index.core.prompts.default_prompt_selectors import (
    DEFAULT_TEXT_QA_PROMPT_SEL,
)
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.types import RESPONSE_TEXT_TYPE


class Accumulate(BaseSynthesizer):
    """Accumulate responses from multiple text chunks."""

    def __init__(
        self,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        prompt_helper: Optional[PromptHelper] = None,
        chat_prompt_helper: Optional[ChatPromptHelper] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        chat_content_qa_template: Optional[BasePromptTemplate] = None,
        output_cls: Optional[Type[BaseModel]] = None,
        streaming: bool = False,
        use_async: bool = False,
        multimodal: bool = False,
    ) -> None:
        super().__init__(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            chat_prompt_helper=chat_prompt_helper,
            streaming=streaming,
            output_cls=output_cls,
            multimodal=multimodal,
        )
        self._text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT_SEL
        self._chat_content_qa_template = (
            chat_content_qa_template or CHAT_CONTENT_QA_PROMPT
        )
        self._use_async = use_async

    def _get_prompts(self) -> PromptDictType:
        return {
            "text_qa_template": self._text_qa_template,
            "chat_content_qa_template": self._chat_content_qa_template,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        if "text_qa_template" in prompts:
            self._text_qa_template = prompts["text_qa_template"]
        if "chat_content_qa_template" in prompts:
            self._chat_content_qa_template = prompts["chat_content_qa_template"]

    def flatten_list(self, md_array: List[List[Any]]) -> List[Any]:
        return [item for sublist in md_array for item in sublist]

    def _format_response(self, outputs: List[Any], separator: str) -> str:
        responses: List[str] = []
        for response in outputs:
            responses.append(response or "Empty Response")

        return separator.join(
            [f"Response {index + 1}: {item}" for index, item in enumerate(responses)]
        )

    def _make_prompt_kwargs(self, chunk: str | ChatMessage) -> dict[str, Any]:
        if self._multimodal:
            template = self._chat_content_qa_template.partial_format(
                query_str="{query_str}"
            )
            return {"context_messages": [chunk], "template": template}
        else:
            template = self._text_qa_template.partial_format(query_str="{query_str}")
            return {"context_str": chunk, "template": template}

    def _give_responses(
        self,
        query_str: str,
        chunk: str | ChatMessage,
        use_async: bool = False,
        **response_kwargs: Any,
    ) -> List[Any]:
        repacked: list[str] | list[ChatMessage]
        if self._multimodal:
            assert isinstance(chunk, ChatMessage)
            template = self._chat_content_qa_template.partial_format(
                query_str=query_str
            )
            repacked = self._chat_prompt_helper.repack(template, [chunk], llm=self._llm)
        else:
            assert isinstance(chunk, str)
            template = self._text_qa_template.partial_format(query_str=query_str)
            repacked = self._prompt_helper.repack(template, [chunk], llm=self._llm)

        predictor: Callable
        if self._output_cls is None:
            predictor = self._llm.apredict if use_async else self._llm.predict
            return [
                predictor(template, **self._make_prompt_kwargs(c), **response_kwargs)
                for c in repacked
            ]
        else:
            predictor = (
                self._llm.astructured_predict
                if use_async
                else self._llm.structured_predict
            )
            return [
                predictor(
                    self._output_cls,
                    template,
                    **self._make_prompt_kwargs(c),
                    **response_kwargs,
                )
                for c in repacked
            ]

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        separator: str = "\n---------------------\n",
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        if self._streaming:
            raise ValueError("Unable to stream in Accumulate response mode")

        tasks = [
            self._give_responses(
                query_str, text_chunk, use_async=True, **response_kwargs
            )
            for text_chunk in text_chunks
        ]

        flattened_tasks = self.flatten_list(tasks)
        outputs = await asyncio.gather(*flattened_tasks)

        return self._format_response(outputs, separator)

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        separator: str = "\n---------------------\n",
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        if self._streaming:
            raise ValueError("Unable to stream in Accumulate response mode")

        tasks = [
            self._give_responses(
                query_str, text_chunk, use_async=self._use_async, **response_kwargs
            )
            for text_chunk in text_chunks
        ]

        outputs = self.flatten_list(tasks)

        if self._use_async:
            outputs = run_async_tasks(outputs)

        return self._format_response(outputs, separator)

    async def aget_response_from_messages(
        self,
        query_str: str,
        message_chunks: Sequence[ChatMessage],
        separator: str = "\n---------------------\n",
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        if self._streaming:
            raise ValueError("Unable to stream in Accumulate response mode")

        tasks = [
            self._give_responses(query_str, chunk, use_async=True, **response_kwargs)
            for chunk in message_chunks
        ]

        flattened_tasks = self.flatten_list(tasks)
        outputs = await asyncio.gather(*flattened_tasks)

        return self._format_response(outputs, separator)

    def get_response_from_messages(
        self,
        query_str: str,
        message_chunks: Sequence[ChatMessage],
        separator: str = "\n---------------------\n",
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        if self._streaming:
            raise ValueError("Unable to stream in Accumulate response mode")

        tasks = [
            self._give_responses(
                query_str, chunk, use_async=self._use_async, **response_kwargs
            )
            for chunk in message_chunks
        ]

        outputs = self.flatten_list(tasks)

        if self._use_async:
            outputs = run_async_tasks(outputs)

        return self._format_response(outputs, separator)
