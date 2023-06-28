"""LLM Prompt Program."""
from llama_index.program.base_program import BasePydanticProgram
from typing import Type, Any, Optional
from llama_index.types import Model
from abc import abstractmethod


class BaseLLMFunctionProgram(BasePydanticProgram):
    """Base LLM Prompt Program.

    This is a base class for LLM endpoints that can return
    a structured output given the prompt.

    NOTE: this only works for structured endpoints atm
    (does not work for text completion endpoints.)

    """

    @classmethod
    @abstractmethod
    def from_defaults(
        cls,
        output_cls: Type[Model],
        prompt_str: str,
        llm: Optional[Any] = None,
        **kwargs: Any,
    ) -> "BaseLLMFunctionProgram":
        """Initialize program from defaults."""
