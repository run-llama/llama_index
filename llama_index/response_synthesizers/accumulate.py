import asyncio
from typing import Any, List, Sequence, Optional

from llama_index.async_utils import run_async_tasks
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT,
)
from llama_index.prompts.prompts import QuestionAnswerPrompt
from llama_index.response_synthesizers.base import BaseSynthesizer
from llama_index.types import RESPONSE_TEXT_TYPE


class Accumulate(BaseSynthesizer):
    """Accumulate responses from multiple text chunks."""

    def __init__(
        self,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        service_context: Optional[ServiceContext] = None,
        streaming: bool = False,
        use_async: bool = False,
    ) -> None:
        super().__init__(
            service_context=service_context,
            streaming=streaming,
        )
        self._text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self._use_async = use_async

    def flatten_list(self, md_array: List[List[Any]]) -> List[Any]:
        return list(item for sublist in md_array for item in sublist)

    def _format_response(self, outputs: List[Any], separator: str) -> str:
        responses: List[str] = []
        for response in outputs:
            responses.append(response or "Empty Response")

        return separator.join(
            [f"Response {index + 1}: {item}" for index, item in enumerate(responses)]
        )

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        separator: str = "\n---------------------\n",
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Apply the same prompt to text chunks and return async responses"""

        if self._streaming:
            raise ValueError("Unable to stream in Accumulate response mode")

        tasks = [
            self._give_responses(query_str, text_chunk, use_async=True)
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
        """Apply the same prompt to text chunks and return responses"""

        if self._streaming:
            raise ValueError("Unable to stream in Accumulate response mode")

        tasks = [
            self._give_responses(query_str, text_chunk, use_async=self._use_async)
            for text_chunk in text_chunks
        ]

        outputs = self.flatten_list(tasks)

        if self._use_async:
            outputs = run_async_tasks(outputs)

        return self._format_response(outputs, separator)

    def _give_responses(
        self, query_str: str, text_chunk: str, use_async: bool = False
    ) -> List[Any]:
        """Give responses given a query and a corresponding text chunk."""
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)

        text_chunks = self._service_context.prompt_helper.repack(
            text_qa_template, [text_chunk]
        )

        predictor = (
            self._service_context.llm_predictor.apredict
            if use_async
            else self._service_context.llm_predictor.predict
        )

        return [
            predictor(
                text_qa_template,
                context_str=cur_text_chunk,
            )
            for cur_text_chunk in text_chunks
        ]
