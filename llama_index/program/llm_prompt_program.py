"""LLM Prompt Program."""
from llama_index.program.base_program import BasePydanticProgram
from llama_index.bridge.langchain import ChatOpenAI
from llama_index.prompts.base import Prompt
from typing import Type, Any, Optional, Union
from llama_index.types import Model
from abc import abstractmethod


class BaseLLMPromptProgram(BasePydanticProgram):
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
    ) -> "BaseLLMPromptProgram":
        """Initialize program from defaults."""

    @property
    @abstractmethod
    def llm(self) -> Any:
        """Get llm."""

    @property
    @abstractmethod
    def prompt_str(self) -> str:
        """Get prompt str."""
