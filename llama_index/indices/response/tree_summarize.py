import asyncio
from typing import Any, List, Optional, Sequence, Tuple

from llama_index.async_utils import run_async_tasks
from llama_index.indices.response.base_builder import BaseResponseBuilder
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.prompt_type import PromptType
from llama_index.prompts.prompts import QuestionAnswerPrompt, SummaryPrompt
from llama_index.token_counter.token_counter import llm_token_counter
from llama_index.types import RESPONSE_TEXT_TYPE


class TreeSummarize(BaseResponseBuilder):
    def __init__(
        self,
        service_context: ServiceContext,
        text_qa_template: QuestionAnswerPrompt,
        streaming: bool = False,
        use_async: bool = True,
    ) -> None:
        super().__init__(
            service_context=service_context,
            streaming=streaming,
        )
        self._text_qa_template = text_qa_template
        self._use_async = use_async

    @llm_token_counter("aget_response")
    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get tree summarize response."""
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        summary_template = SummaryPrompt.from_prompt(
            text_qa_template, prompt_type=PromptType.SUMMARY
        )

        # repack text_chunks so that each chunk fills the context window
        text_chunks = self._service_context.prompt_helper.repack(
            summary_template, text_chunks=text_chunks
        )

        # give final response if there is only one chunk
        if len(text_chunks) == 1:
            if self._streaming:
                response, _ = self._service_context.llm_predictor.stream(
                    summary_template,
                    context_str=text_chunks[0],
                )
            else:
                response, _ = await self._service_context.llm_predictor.apredict(
                    summary_template,
                    context_str=text_chunks[0],
                )
            return response

        else:
            # summarize each chunk
            tasks = [
                self._service_context.llm_predictor.apredict(
                    summary_template,
                    context_str=text_chunk,
                )
                for text_chunk in text_chunks
            ]

            outputs: List[Tuple[str, str]] = await asyncio.gather(*tasks)
            summaries = [output[0] for output in outputs]

            # recursively summarize the summaries
            return await self.aget_response(
                query_str=query_str,
                text_chunks=summaries,
            )

    @llm_token_counter("get_response")
    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get tree summarize response."""
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        summary_template = SummaryPrompt.from_prompt(
            text_qa_template, prompt_type=PromptType.SUMMARY
        )

        # repack text_chunks so that each chunk fills the context window
        text_chunks = self._service_context.prompt_helper.repack(
            summary_template, text_chunks=text_chunks
        )

        # give final response if there is only one chunk
        if len(text_chunks) == 1:
            if self._streaming:
                response, _ = self._service_context.llm_predictor.stream(
                    summary_template,
                    context_str=text_chunks[0],
                )
            else:
                response, _ = self._service_context.llm_predictor.predict(
                    summary_template,
                    context_str=text_chunks[0],
                )
            return response

        else:
            # summarize each chunk
            if self._use_async:
                tasks = [
                    self._service_context.llm_predictor.apredict(
                        summary_template,
                        context_str=text_chunk,
                    )
                    for text_chunk in text_chunks
                ]

                outputs: List[Tuple[str, str]] = run_async_tasks(tasks)
                summaries = [output[0] for output in outputs]
            else:
                summaries = [
                    self._service_context.llm_predictor.predict(
                        summary_template,
                        context_str=text_chunk,
                    )[0]
                    for text_chunk in text_chunks
                ]

            # recursively summarize the summaries
            return self.get_response(
                query_str=query_str,
                text_chunks=summaries,
            )
