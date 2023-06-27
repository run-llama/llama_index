import asyncio
from typing import Any, List, Optional, Sequence, Tuple

from llama_index.async_utils import run_async_tasks
from llama_index.indices.response.base_builder import BaseResponseBuilder
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.prompts.prompt_type import PromptType
from llama_index.prompts.prompts import QuestionAnswerPrompt, SummaryPrompt
from llama_index.types import RESPONSE_TEXT_TYPE


class TreeSummarize(BaseResponseBuilder):
    """
    Tree summarize response builder.

    This response builder recursively merges text chunks and summarizes them
    in a bottom-up fashion (i.e. building a tree from leaves to root).

    More concretely, at each recursively step:
    1. we repack the text chunks so that each chunk fills the context window of the LLM
    2. if there is only one chunk, we give the final response
    3. otherwise, we summarize each chunk and recursively summarize the summaries.
    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        streaming: bool = False,
        use_async: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            service_context=service_context,
            streaming=streaming,
        )
        self._text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self._use_async = use_async
        self._verbose = verbose

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
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

        if self._verbose:
            print(f"{len(text_chunks)} text chunks after repacking")

        # give final response if there is only one chunk
        if len(text_chunks) == 1:
            response: RESPONSE_TEXT_TYPE
            if self._streaming:
                response = self._service_context.llm_predictor.stream(
                    summary_template,
                    context_str=text_chunks[0],
                )
            else:
                response = await self._service_context.llm_predictor.apredict(
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

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
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

        if self._verbose:
            print(f"{len(text_chunks)} text chunks after repacking")

        # give final response if there is only one chunk
        if len(text_chunks) == 1:
            response: RESPONSE_TEXT_TYPE
            if self._streaming:
                response = self._service_context.llm_predictor.stream(
                    summary_template,
                    context_str=text_chunks[0],
                )
            else:
                response = self._service_context.llm_predictor.predict(
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
