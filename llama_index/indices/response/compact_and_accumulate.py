from typing import Any, Sequence

from llama_index.indices.response.accumulate import Accumulate
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.prompts import QuestionAnswerPrompt
from llama_index.types import RESPONSE_TEXT_TYPE
from llama_index.utils import temp_set_attrs


class CompactAndAccumulate(Accumulate):
    def __init__(
        self,
        service_context: ServiceContext,
        text_qa_template: QuestionAnswerPrompt,
        streaming: bool = False,
        use_async: bool = False,
    ) -> None:
        super().__init__(
            service_context=service_context,
            text_qa_template=text_qa_template,
            streaming=streaming,
            use_async=use_async,
        )
        self.text_qa_template = text_qa_template

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        separator: str = "\n---------------------\n",
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        return self.get_response(query_str, text_chunks, separator, use_aget=True)

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        separator: str = "\n---------------------\n",
        use_aget: bool = False,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get compact response."""
        # use prompt helper to fix compact text_chunks under the prompt limitation
        text_qa_template = self.text_qa_template.partial_format(query_str=query_str)

        with temp_set_attrs(self._service_context.prompt_helper):
            new_texts = self._service_context.prompt_helper.repack(
                text_qa_template, text_chunks
            )

            responder = super().aget_response if use_aget else super().get_response
            response = responder(
                query_str=query_str, text_chunks=new_texts, separator=separator
            )
        return response
