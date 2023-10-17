from typing import Optional

from llama_index.indices.service_context import ServiceContext
from llama_index.selectors.llm_selectors import LLMMultiSelector, LLMSingleSelector
from llama_index.selectors.pydantic_selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.selectors.types import BaseSelector


def get_selector_from_context(
    service_context: ServiceContext, is_multi: bool = False
) -> BaseSelector:
    """Get a selector from a service context. Prefers Pydantic selectors if possible."""
    selector: Optional[BaseSelector] = None

    if is_multi:
        try:
            llm = service_context.llm_predictor.llm
            selector = PydanticMultiSelector.from_defaults(llm=llm)  # type: ignore
        except ValueError:
            selector = LLMMultiSelector.from_defaults(service_context=service_context)
    else:
        try:
            llm = service_context.llm_predictor.llm
            selector = PydanticSingleSelector.from_defaults(llm=llm)  # type: ignore
        except ValueError:
            selector = LLMSingleSelector.from_defaults(service_context=service_context)

    assert selector is not None

    return selector
