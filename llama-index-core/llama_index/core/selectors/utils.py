from typing import Optional

from llama_index.core.base.base_selector import BaseSelector
from llama_index.core.llms.llm import LLM
from llama_index.core.selectors.llm_selectors import (
    LLMMultiSelector,
    LLMSingleSelector,
)
from llama_index.core.selectors.pydantic_selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)


def get_selector_from_llm(llm: LLM, is_multi: bool = False) -> BaseSelector:
    """Get a selector from a service context. Prefers Pydantic selectors if possible."""
    selector: Optional[BaseSelector] = None

    if is_multi:
        try:
            selector = PydanticMultiSelector.from_defaults(llm=llm)  # type: ignore
        except ValueError:
            selector = LLMMultiSelector.from_defaults(llm=llm)
    else:
        try:
            selector = PydanticSingleSelector.from_defaults(llm=llm)  # type: ignore
        except ValueError:
            selector = LLMSingleSelector.from_defaults(llm=llm)

    assert selector is not None

    return selector
