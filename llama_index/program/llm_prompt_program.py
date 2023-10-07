"""LLM Prompt Program."""
from abc import abstractmethod
from typing import Any, Generic, Optional, Type, TypeVar

from llama_index.bridge.pydantic import BaseModel
from llama_index.prompts.base import PromptTemplate
from llama_index.types import BasePydanticProgram, Model

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
        prompt: Optional[PromptTemplate] = None,
        llm: Optional[LM] = None,
        **kwargs: Any,
    ) -> "BaseLLMFunctionProgram":
        """Initialize program from defaults."""
