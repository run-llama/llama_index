from typing import Any, Optional, Sequence
from llama_index.prompts.utils import get_biggest_prompt

from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.indices.response.refine import Refine
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.base import Prompt
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.types import RESPONSE_TEXT_TYPE


CITATION_QA_TEMPLATE = Prompt(
    "Please provide an answer based on the given sources. "
    "When referencing information from a source, cite the appropriate source(s). "
    "For example:\n"
    "Source 1:\n"
    "The sky is red.\n"
    "Source 2:\n"
    "Water is wet.\n"
    "Query: What color is the sky?\n"
    "Answer: According to [1], the sky is red.\n"
    "Now it's your turn. Below are several numbered sources of information:"
    "\n------\n"
    "{context_str}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)



class Citation(Refine):
    def __init__(
        self,
        service_context: ServiceContext,
        text_qa_template: QuestionAnswerPrompt,
        streaming: bool = False,
    ) -> None:
        super().__init__(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=Prompt('{query_str}'),
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

    def _get_source_chunks(self, text_chunks: Sequence[str]) -> Sequence[str]:
        text_splitter = TokenTextSplitter(
            chunk_size=256,
            chunk_overlap=20
        )

        new_text_chunks = text_splitter.split_text_with_overlaps(" ".join(text_chunks))
        return [f"Source {i}:\n{str(x).strip()}\n" for i, x in enumerate(new_text_chunks)]

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
        #text_qa_template = self.text_qa_template.partial_format(query_str=query_str)
        text_qa_template = CITATION_QA_TEMPLATE.partial_format(query_str=query_str)
        self.text_qa_template = text_qa_template
        refine_template = self._refine_template.partial_format(query_str=query_str)

        max_prompt = get_biggest_prompt([text_qa_template, refine_template])

        source_text_chunks = self._get_source_chunks(text_chunks)
        new_texts = self._service_context.prompt_helper.repack(max_prompt, source_text_chunks)
        response = super().get_response(
            query_str=query_str, text_chunks=new_texts, prev_response=prev_response
        )
        return response
