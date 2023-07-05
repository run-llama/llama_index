from typing import Optional

from llama_index.callbacks.base import CallbackManager
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
from llama_index.response_synthesizers.accumulate import Accumulate
from llama_index.response_synthesizers.base import BaseSynthesizer
from llama_index.response_synthesizers.compact_and_accumulate import (
    CompactAndAccumulate,
)
from llama_index.response_synthesizers.compact_and_refine import CompactAndRefine
from llama_index.response_synthesizers.generation import Generation
from llama_index.response_synthesizers.refine import Refine
from llama_index.response_synthesizers.simple_summarize import SimpleSummarize
from llama_index.response_synthesizers.tree_summarize import TreeSummarize
from llama_index.response_synthesizers.type import ResponseMode


def get_response_synthesizer(
    service_context: Optional[ServiceContext] = None,
    text_qa_template: Optional[QuestionAnswerPrompt] = None,
    refine_template: Optional[RefinePrompt] = None,
    simple_template: Optional[SimpleInputPrompt] = None,
    response_mode: ResponseMode = ResponseMode.COMPACT,
    use_async: bool = False,
    streaming: bool = False,
) -> BaseSynthesizer:
    """Get a response synthesizer."""

    text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
    refine_template = refine_template or DEFAULT_REFINE_PROMPT_SEL
    simple_template = simple_template or DEFAULT_SIMPLE_INPUT_PROMPT

    service_context = (
        service_context or ServiceContext.get_global() or ServiceContext.from_defaults()
    )

    if response_mode == ResponseMode.REFINE:
        return Refine(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            streaming=streaming,
        )
    elif response_mode == ResponseMode.COMPACT:
        return CompactAndRefine(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            streaming=streaming,
        )
    elif response_mode == ResponseMode.TREE_SUMMARIZE:
        return TreeSummarize(
            service_context=service_context,
            text_qa_template=text_qa_template,
            streaming=streaming,
            use_async=use_async,
        )
    elif response_mode == ResponseMode.SIMPLE_SUMMARIZE:
        return SimpleSummarize(
            service_context=service_context,
            text_qa_template=text_qa_template,
            streaming=streaming,
        )
    elif response_mode == ResponseMode.GENERATION:
        return Generation(
            service_context=service_context,
            simple_template=simple_template,
            streaming=streaming,
        )
    elif response_mode == ResponseMode.ACCUMULATE:
        return Accumulate(
            service_context=service_context,
            text_qa_template=text_qa_template,
            streaming=streaming,
            use_async=use_async,
        )
    elif response_mode == ResponseMode.COMPACT_ACCUMULATE:
        return CompactAndAccumulate(
            service_context=service_context,
            text_qa_template=text_qa_template,
            streaming=streaming,
            use_async=use_async,
        )
    else:
        raise ValueError(f"Unknown mode: {response_mode}")
