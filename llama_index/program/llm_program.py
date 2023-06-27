from typing import Any, Dict, Generic, Optional, Type, Union

from llama_index.bridge.langchain import ChatOpenAI, HumanMessage

from llama_index.types import Model
from llama_index.program.base_program import BasePydanticProgram
from llama_index.prompts.base import Prompt
from llama_index.bridge.langchain import BaseLanguageModel
from llama_index.output_parsers.pydantic import PydanticOutputParser


SUPPORTED_MODEL_NAMES = [
    "gpt-3.5-turbo-0613",
    "gpt-4-0613",
]


class LLMTextCompletionProgram(BasePydanticProgram, Generic[Model]):
    """
    LLM Text Completion Program.

    Uses generic LLM text completion + an output parser to generate a structured output.

    """

    def __init__(
        self,
        output_parser: PydanticOutputParser,
        prompt: Prompt,
        llm: BaseLanguageModel,
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
        llm: Optional[BaseLanguageModel] = None,
        verbose: bool = False,
        function_call: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "LLMTextCompletionProgram":
        llm = llm or ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
        if not isinstance(llm, ChatOpenAI):
            raise ValueError("llm must be a ChatOpenAI instance")

        if llm.model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError(
                f"Model name {llm.model_name} not supported. "
                f"Supported model names: {SUPPORTED_MODEL_NAMES}"
            )
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
    def output_parser(self) -> Type[Model]:
        return self._output_parser.output_cls

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Model:
        prompt_with_parse_instrs_tmpl = self._output_parser.format(
            self._prompt.template_str
        )
        prompt_with_parse_instrs = Prompt(prompt_with_parse_instrs_tmpl)

        formatted_prompt = prompt_with_parse_instrs.format(**kwargs)

        raw_output = self._llm.predict(prompt=formatted_prompt)
        model_output = self._output_parser.parse(raw_output)
        return model_output
