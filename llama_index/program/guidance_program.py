from typing import TYPE_CHECKING, Any, Optional, Type, cast

from llama_index.bridge.pydantic import BaseModel
from llama_index.program.llm_prompt_program import BaseLLMFunctionProgram
from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.guidance_utils import (
    parse_pydantic_from_guidance_program,
    pydantic_to_guidance_output_template_markdown,
)

if TYPE_CHECKING:
    from guidance.llms import LLM as GuidanceLLM


class GuidancePydanticProgram(BaseLLMFunctionProgram["GuidanceLLM"]):
    """
    A guidance-based function that returns a pydantic model.

    Note: this interface is not yet stable.
    """

    def __init__(
        self,
        output_cls: Type[BaseModel],
        prompt_template_str: str,
        guidance_llm: Optional["GuidanceLLM"] = None,
        verbose: bool = False,
    ):
        try:
            from guidance import Program
            from guidance.llms import OpenAI
        except ImportError as e:
            raise ImportError(
                "guidance package not found." "please run `pip install guidance`"
            ) from e

        llm = guidance_llm or OpenAI("text-davinci-003")
        output_str = pydantic_to_guidance_output_template_markdown(output_cls)
        full_str = prompt_template_str + "\n" + output_str
        self._full_str = full_str
        self._guidance_program = Program(full_str, llm=llm, silent=not verbose)
        self._output_cls = output_cls
        self._verbose = verbose

    @classmethod
    def from_defaults(
        cls,
        output_cls: Type[BaseModel],
        prompt_template_str: Optional[str] = None,
        prompt: Optional[PromptTemplate] = None,
        llm: Optional["GuidanceLLM"] = None,
        **kwargs: Any,
    ) -> "BaseLLMFunctionProgram":
        """From defaults."""
        if prompt is None and prompt_template_str is None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt is not None and prompt_template_str is not None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt is not None:
            prompt_template_str = prompt.template
        prompt_template_str = cast(str, prompt_template_str)
        return cls(
            output_cls,
            prompt_template_str,
            guidance_llm=llm,
            **kwargs,
        )

    @property
    def output_cls(self) -> Type[BaseModel]:
        return self._output_cls

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> BaseModel:
        executed_program = self._guidance_program(**kwargs)

        return parse_pydantic_from_guidance_program(
            program=executed_program, cls=self._output_cls
        )
