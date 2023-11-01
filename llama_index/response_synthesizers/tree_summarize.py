import asyncio
from typing import Any, List, Optional, Sequence

from llama_index.async_utils import run_async_tasks
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts import BasePromptTemplate
from llama_index.prompts.default_prompt_selectors import (
    DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
)
from llama_index.prompts.mixin import PromptDictType
from llama_index.response_synthesizers.base import BaseSynthesizer
from llama_index.types import RESPONSE_TEXT_TYPE, BaseModel


class TreeSummarize(BaseSynthesizer):
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
        summary_template: Optional[BasePromptTemplate] = None,
        service_context: Optional[ServiceContext] = None,
        output_cls: Optional[BaseModel] = None,
        streaming: bool = False,
        use_async: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            service_context=service_context, streaming=streaming, output_cls=output_cls
        )
        self._summary_template = summary_template or DEFAULT_TREE_SUMMARIZE_PROMPT_SEL
        self._use_async = use_async
        self._verbose = verbose

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"summary_template": self._summary_template}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "summary_template" in prompts:
            self._summary_template = prompts["summary_template"]

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get tree summarize response."""
        summary_template = self._summary_template.partial_format(query_str=query_str)
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
                    summary_template, context_str=text_chunks[0], **response_kwargs
                )
            else:
                response = await self._service_context.llm_predictor.apredict(
                    summary_template,
                    output_cls=self._output_cls,
                    context_str=text_chunks[0],
                    **response_kwargs,
                )

            # return pydantic object if output_cls is specified
            return (
                response
                if self._output_cls is None
                else self._output_cls.parse_raw(response)
            )

        else:
            # summarize each chunk
            tasks = [
                self._service_context.llm_predictor.apredict(
                    summary_template,
                    output_cls=self._output_cls,
                    context_str=text_chunk,
                    **response_kwargs,
                )
                for text_chunk in text_chunks
            ]

            summaries: List[str] = await asyncio.gather(*tasks)

            # recursively summarize the summaries
            return await self.aget_response(
                query_str=query_str,
                text_chunks=summaries,
                **response_kwargs,
            )

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get tree summarize response."""
        summary_template = self._summary_template.partial_format(query_str=query_str)
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
                    summary_template, context_str=text_chunks[0], **response_kwargs
                )
            else:
                response = self._service_context.llm_predictor.predict(
                    summary_template,
                    output_cls=self._output_cls,
                    context_str=text_chunks[0],
                    **response_kwargs,
                )

            # return pydantic object if output_cls is specified
            return (
                response
                if self._output_cls is None
                else self._output_cls.parse_raw(response)
            )

        else:
            # summarize each chunk
            if self._use_async:
                tasks = [
                    self._service_context.llm_predictor.apredict(
                        summary_template,
                        output_cls=self._output_cls,
                        context_str=text_chunk,
                        **response_kwargs,
                    )
                    for text_chunk in text_chunks
                ]

                summaries: List[str] = run_async_tasks(tasks)
            else:
                summaries = [
                    self._service_context.llm_predictor.predict(
                        summary_template,
                        output_cls=self._output_cls,
                        context_str=text_chunk,
                        **response_kwargs,
                    )
                    for text_chunk in text_chunks
                ]

            # recursively summarize the summaries
            return self.get_response(
                query_str=query_str, text_chunks=summaries, **response_kwargs
            )
