from typing import Optional

from llama_index.indices.response.accumulate import Accumulate
from llama_index.indices.response.base_builder import BaseResponseBuilder
from llama_index.indices.response.compact_and_accumulate import CompactAndAccumulate
from llama_index.indices.response.compact_and_refine import CompactAndRefine
from llama_index.indices.response.generation import Generation
from llama_index.indices.response.refine import Refine
from llama_index.indices.response.simple_summarize import SimpleSummarize
from llama_index.indices.response.tree_summarize import TreeSummarize
from llama_index.indices.response.type import ResponseMode
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.default_prompt_selectors import DEFAULT_REFINE_PROMPT_SEL
from llama_index.prompts.default_prompts import (
    DEFAULT_SIMPLE_INPUT_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)
from llama_index.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    SimpleInputPrompt,
)


def get_response_builder(
    service_context: ServiceContext,
    text_qa_template: Optional[QuestionAnswerPrompt] = None,
    refine_template: Optional[RefinePrompt] = None,
    simple_template: Optional[SimpleInputPrompt] = None,
    mode: ResponseMode = ResponseMode.COMPACT,
    use_async: bool = False,
    streaming: bool = False,
) -> BaseResponseBuilder:
    text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
    refine_template = refine_template or DEFAULT_REFINE_PROMPT_SEL
    simple_template = simple_template or DEFAULT_SIMPLE_INPUT_PROMPT
    if mode == ResponseMode.REFINE:
        return Refine(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            streaming=streaming,
        )
    elif mode == ResponseMode.COMPACT:
        return CompactAndRefine(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            streaming=streaming,
        )
    elif mode == ResponseMode.TREE_SUMMARIZE:
        return TreeSummarize(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            streaming=streaming,
            use_async=use_async,
        )
    elif mode == ResponseMode.SIMPLE_SUMMARIZE:
        return SimpleSummarize(
            service_context=service_context,
            text_qa_template=text_qa_template,
            streaming=streaming,
        )
    elif mode == ResponseMode.GENERATION:
        return Generation(
            service_context=service_context,
            simple_template=simple_template,
            streaming=streaming,
        )
    elif mode == ResponseMode.ACCUMULATE:
        return Accumulate(
            service_context=service_context,
            text_qa_template=text_qa_template,
            streaming=streaming,
            use_async=use_async,
        )
    elif mode == ResponseMode.COMPACT_ACCUMULATE:
        return CompactAndAccumulate(
            service_context=service_context,
            text_qa_template=text_qa_template,
            streaming=streaming,
            use_async=use_async,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
