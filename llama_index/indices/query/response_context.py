from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from llama_index.callbacks.base import CallbackManager
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.indices.response import (
    BaseResponseBuilder,
    ResponseMode,
)
from llama_index.optimization.optimizer import BaseTokenUsageOptimizer
from llama_index.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    SimpleInputPrompt,
)
from llama_index.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT,
    DEFAULT_SIMPLE_INPUT_PROMPT,
)
from llama_index.prompts.default_prompt_selectors import DEFAULT_REFINE_PROMPT_SEL


class ResponseContext(BaseModel):
    """Response context, used to hold various response-related objects and services."""

    # pre-processing
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = Field(
        default=None, description="A list of node postprocessors."
    )
    optimizer: Optional[BaseTokenUsageOptimizer] = Field(
        default=None, description="A token usage optimizer."
    )

    # response-builder
    response_builder: Optional[BaseResponseBuilder] = Field(
        default=None,
        description="A response builder that uses text chunks to build a response.",
    )
    text_qa_template: QuestionAnswerPrompt = Field(
        default=DEFAULT_TEXT_QA_PROMPT,
        description="The prompt template used for the initial QA LLM call.",
    )
    refine_template: RefinePrompt = Field(
        default=DEFAULT_REFINE_PROMPT_SEL,
        description="The prompt template used for the refine LLM call(s).",
    )
    simple_template: SimpleInputPrompt = Field(
        default=DEFAULT_SIMPLE_INPUT_PROMPT,
        description=(
            "The prompt template used for simple generation, when "
            "response_mode='generation'."
        ),
    )
    response_mode: ResponseMode = Field(
        default=ResponseMode.COMPACT,
        description="The response mode, used to fetch a response builder.",
    )
    response_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs passed to the response builder.",
    )

    # miscelaneous
    callback_manager: Optional[CallbackManager] = Field(
        default=None, description="A callback manager."
    )
    streaming: bool = Field(
        default=False, description="Whether to stream the response."
    )
    use_async: bool = Field(default=False, description="Whether to use async.")
    verbose: bool = Field(default=False, description="Whether to print verbose output.")

    # ignore pydantic's type-checking for arbitrary types
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_defaults(
        cls,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        optimizer: Optional[BaseTokenUsageOptimizer] = None,
        response_mode: ResponseMode = ResponseMode.COMPACT,
        text_qa_template: Optional[QuestionAnswerPrompt] = DEFAULT_TEXT_QA_PROMPT,
        refine_template: Optional[RefinePrompt] = DEFAULT_REFINE_PROMPT_SEL,
        simple_template: Optional[SimpleInputPrompt] = DEFAULT_SIMPLE_INPUT_PROMPT,
        response_kwargs: Optional[Dict] = Field(default_factory=dict),
        callback_manager: Optional[CallbackManager] = None,
        streaming: bool = False,
        use_async: bool = False,
        verbose: bool = False,
    ) -> "ResponseContext":
        return cls(
            node_postprocessors=node_postprocessors,
            optimizer=optimizer,
            response_mode=response_mode,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            simple_template=simple_template,
            response_kwargs=response_kwargs,
            callback_manager=callback_manager,
            streaming=streaming,
            use_async=use_async,
            verbose=verbose,
        )
