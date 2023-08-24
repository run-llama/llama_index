from typing import Any, Dict, Optional, Type, Union

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel

from llama_index.llms.base import LLM
from llama_index.llms.openai import OpenAI
from llama_index.output_parsers.pydantic import PydanticOutputParser
from llama_index.program.base_program import BasePydanticProgram
from llama_index.prompts.base import Prompt


class LLMTextCompletionProgram(BasePydanticProgram[BaseModel]):
    """
    LLM Text Completion Program.

    Uses generic LLM text completion + an output parser to generate a structured output.

    """

    def __init__(
        self,
        output_parser: PydanticOutputParser,
        prompt: Prompt,
        llm: LLM,
        function_call: Union[str, Dict[str, Any]],
        verbose: bool = False,
    ) -> None:
        self._output_parser = output_parser
        self._llm = llm
        self._prompt = prompt
        self._verbose = verbose
        self._function_call = function_call

    @classmethod
    def from_defaults(
        cls,
        output_parser: PydanticOutputParser,
        prompt_template_str: str,
        llm: Optional[LLM] = None,
        verbose: bool = False,
        function_call: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "LLMTextCompletionProgram":
        llm = llm or OpenAI(temperature=0, model="gpt-3.5-turbo-0613")
        prompt = Prompt(prompt_template_str)
        function_call = function_call or {
            "name": output_parser.output_cls.schema()["title"]
        }
        return cls(
            output_parser,
            prompt=prompt,
            llm=llm,
            function_call=function_call,
            verbose=verbose,
        )

    @property
    def output_cls(self) -> Type[BaseModel]:
        return self._output_parser.output_cls

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> BaseModel:
        prompt_with_parse_instrs_tmpl = self._output_parser.format(
            self._prompt.original_template
        )
        prompt_with_parse_instrs = Prompt(prompt_with_parse_instrs_tmpl)

        formatted_prompt = prompt_with_parse_instrs.format(**kwargs)

        response = self._llm.complete(formatted_prompt)
        raw_output = response.text
        model_output = self._output_parser.parse(raw_output)
        return model_output
