from typing import Optional

from llama_index.callbacks.base import CallbackManager
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.default_prompt_selectors import DEFAULT_REFINE_PROMPT_SEL
from llama_index.prompts.default_prompts import (
    DEFAULT_SIMPLE_INPUT_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
    DEFAULT_CUSTOM_QUERY_PROMPT,
    DEFAULT_CUSTOM_REFINE_PROMPT,
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
from llama_index.response_synthesizers.no_text import NoText


def get_response_synthesizer(
    service_context: Optional[ServiceContext] = None,
    text_qa_template: Optional[QuestionAnswerPrompt] = None,
    refine_template: Optional[RefinePrompt] = None,
    simple_template: Optional[SimpleInputPrompt] = None,
    response_mode: ResponseMode = ResponseMode.COMPACT,
    callback_manager: Optional[CallbackManager] = None,
    system_prompt: Optional[str] = None,
    post_query_prompt: Optional[str] = None,
    query_wrapper_prompt: Optional[SimpleInputPrompt] = None,
    use_async: bool = False,
    streaming: bool = False,
) -> BaseSynthesizer:
    """Get a response synthesizer."""

    if system_prompt or post_query_prompt:
        # Change unused custom fields to empty strings for formatting
        system_prompt = system_prompt if system_prompt else ""
        post_query_prompt = post_query_prompt if post_query_prompt else ""

        text_qa_template = text_qa_template or DEFAULT_CUSTOM_QUERY_PROMPT
        text_qa_template = text_qa_template.partial_format(
            system_prompt=system_prompt, post_query_prompt=post_query_prompt
        )
        refine_template = refine_template or DEFAULT_CUSTOM_REFINE_PROMPT
        refine_template = refine_template.partial_format(
            post_query_prompt=post_query_prompt
        )
    else:
        text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        refine_template = refine_template or DEFAULT_REFINE_PROMPT_SEL
    simple_template = simple_template or DEFAULT_SIMPLE_INPUT_PROMPT

    service_context = service_context or ServiceContext.from_defaults(
        callback_manager=callback_manager
    )

    if response_mode == ResponseMode.REFINE:
        return Refine(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            streaming=streaming,
            query_wrapper_prompt=query_wrapper_prompt,
        )
    elif response_mode == ResponseMode.COMPACT:
        return CompactAndRefine(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            streaming=streaming,
            query_wrapper_prompt=query_wrapper_prompt,
        )
    elif response_mode == ResponseMode.TREE_SUMMARIZE:
        return TreeSummarize(
            service_context=service_context,
            text_qa_template=text_qa_template,
            streaming=streaming,
            query_wrapper_prompt=query_wrapper_prompt,
            use_async=use_async,
        )
    elif response_mode == ResponseMode.SIMPLE_SUMMARIZE:
        return SimpleSummarize(
            service_context=service_context,
            text_qa_template=text_qa_template,
            streaming=streaming,
            query_wrapper_prompt=query_wrapper_prompt,
        )
    elif response_mode == ResponseMode.GENERATION:
        return Generation(
            service_context=service_context,
            simple_template=simple_template,
            streaming=streaming,
            query_wrapper_prompt=query_wrapper_prompt,
        )
    elif response_mode == ResponseMode.ACCUMULATE:
        return Accumulate(
            service_context=service_context,
            text_qa_template=text_qa_template,
            streaming=streaming,
            query_wrapper_prompt=query_wrapper_prompt,
            use_async=use_async,
        )
    elif response_mode == ResponseMode.COMPACT_ACCUMULATE:
        return CompactAndAccumulate(
            service_context=service_context,
            text_qa_template=text_qa_template,
            streaming=streaming,
            query_wrapper_prompt=query_wrapper_prompt,
            use_async=use_async,
        )
    elif response_mode == ResponseMode.NO_TEXT:
        return NoText(
            service_context=service_context,
            streaming=streaming,
        )
    else:
        raise ValueError(f"Unknown mode: {response_mode}")
