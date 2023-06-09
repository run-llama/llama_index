from typing import TYPE_CHECKING, Any, Generic, Optional, Type

from llama_index.program.base_program import BasePydanticProgram, Model
from llama_index.prompts.guidance_utils import (
    parse_pydantic_from_guidance_program,
    pydantic_to_guidance_output_template_markdown,
)

if TYPE_CHECKING:
    from guidance.llms import LLM as GuidanceLLM


class GuidancePydanticProgram(BasePydanticProgram, Generic[Model]):
    def __init__(
        self,
        output_cls: Type[Model],
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
        self._guidance_program = Program(full_str, llm=llm, silent=not verbose)
        self._output_cls = output_cls
        self._verbose = verbose

    @property
    def output_cls(self) -> Type[Model]:
        return self._output_cls

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Model:
        executed_program = self._guidance_program(**kwargs)

        pydantic_obj = parse_pydantic_from_guidance_program(
            program=executed_program, cls=self._output_cls, verbose=self._verbose
        )
        return pydantic_obj
