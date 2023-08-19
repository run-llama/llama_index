"""LLM Prompt Program."""
from llama_index.program.base_program import BasePydanticProgram
from typing import Type, Any, Optional, TypeVar, Generic
from llama_index.types import Model
from llama_index.prompts.base import Prompt
from abc import abstractmethod
from pydantic import BaseModel


LM = TypeVar("LM")


class BaseLLMFunctionProgram(BasePydanticProgram[BaseModel], Generic[LM]):
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
        prompt_template_str: Optional[str] = None,
        prompt: Optional[Prompt] = None,
        llm: Optional[LM] = None,
        **kwargs: Any,
    ) -> "BaseLLMFunctionProgram":
        """Initialize program from defaults."""
