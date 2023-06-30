from typing import List, Optional

from llama_index.callbacks.base import CallbackManager
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
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
from llama_index.synthesizers.accumulate import Accumulate
from llama_index.synthesizers.base import BaseSynthesizer
from llama_index.synthesizers.compact_and_accumulate import CompactAndAccumulate
from llama_index.synthesizers.compact_and_refine import CompactAndRefine
from llama_index.synthesizers.generation import Generation
from llama_index.synthesizers.refine import Refine
from llama_index.synthesizers.simple_summarize import SimpleSummarize
from llama_index.synthesizers.tree_summarize import TreeSummarize
from llama_index.synthesizers.type import ResponseMode


def get_response_synthesizer(
    service_context: Optional[ServiceContext] = None,
    text_qa_template: Optional[QuestionAnswerPrompt] = DEFAULT_TEXT_QA_PROMPT,
    refine_template: Optional[RefinePrompt] = DEFAULT_REFINE_PROMPT_SEL,
    simple_template: Optional[SimpleInputPrompt] = DEFAULT_SIMPLE_INPUT_PROMPT,
    mode: ResponseMode = ResponseMode.COMPACT,
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    callback_manager: Optional[CallbackManager] = None,
    use_async: bool = False,
    streaming: bool = False,
) -> BaseSynthesizer:
    """Get a response synthesizer."""

    service_context = service_context or ServiceContext.from_defaults(callback_manager=callback_manager)

    if mode == ResponseMode.REFINE:
        return Refine(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            node_postprocessors=node_postprocessors,
            streaming=streaming,
        )
    elif mode == ResponseMode.COMPACT:
        return CompactAndRefine(
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            node_postprocessors=node_postprocessors,
            streaming=streaming,
        )
    elif mode == ResponseMode.TREE_SUMMARIZE:
        return TreeSummarize(
            service_context=service_context,
            text_qa_template=text_qa_template,
            node_postprocessors=node_postprocessors,
            streaming=streaming,
            use_async=use_async,
        )
    elif mode == ResponseMode.SIMPLE_SUMMARIZE:
        return SimpleSummarize(
            service_context=service_context,
            text_qa_template=text_qa_template,
            node_postprocessors=node_postprocessors,
            streaming=streaming,
        )
    elif mode == ResponseMode.GENERATION:
        return Generation(
            service_context=service_context,
            simple_template=simple_template,
            node_postprocessors=node_postprocessors,
            streaming=streaming,
        )
    elif mode == ResponseMode.ACCUMULATE:
        return Accumulate(
            service_context=service_context,
            text_qa_template=text_qa_template,
            node_postprocessors=node_postprocessors,
            streaming=streaming,
            use_async=use_async,
        )
    elif mode == ResponseMode.COMPACT_ACCUMULATE:
        return CompactAndAccumulate(
            service_context=service_context,
            text_qa_template=text_qa_template,
            node_postprocessors=node_postprocessors,
            streaming=streaming,
            use_async=use_async,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
