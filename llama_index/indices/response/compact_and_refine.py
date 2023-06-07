from typing import Any, Optional, Sequence
from llama_index.prompts.utils import get_biggest_prompt

from llama_index.indices.response.refine import Refine
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.types import RESPONSE_TEXT_TYPE


class CompactAndRefine(Refine):
    def __init__(
        self,
        service_context: ServiceContext,
        text_qa_template: QuestionAnswerPrompt,
        refine_template: RefinePrompt,
        streaming: bool = False,
    ) -> None:
        super().__init__(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            streaming=streaming,
        )

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        return self.get_response(query_str, text_chunks, prev_response)

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[str] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get compact response."""
        # use prompt helper to fix compact text_chunks under the prompt limitation
        # TODO: This is a temporary fix - reason it's temporary is that
        # the refine template does not account for size of previous answer.
        text_qa_template = self.text_qa_template.partial_format(query_str=query_str)
        refine_template = self._refine_template.partial_format(query_str=query_str)

        max_prompt = get_biggest_prompt([text_qa_template, refine_template])
        new_texts = self._service_context.prompt_helper.repack(max_prompt, text_chunks)
        response = super().get_response(
            query_str=query_str, text_chunks=new_texts, prev_response=prev_response
        )
        return response
