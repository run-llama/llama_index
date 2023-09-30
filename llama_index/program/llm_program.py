from typing import Any, List, Optional, Type, cast

from llama_index.bridge.pydantic import BaseModel
from llama_index.llms.base import LLM, ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.output_parsers.pydantic import PydanticOutputParser
from llama_index.prompts.base import PromptTemplate
from llama_index.types import BasePydanticProgram


class LLMTextCompletionProgram(BasePydanticProgram[BaseModel]):
    """
    LLM Text Completion Program.

    Uses generic LLM text completion + an output parser to generate a structured output.

    """

    def __init__(
        self,
        output_parser: PydanticOutputParser,
        llm: LLM,
        prompt: Optional[PromptTemplate] = None,
        messages: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
    ) -> None:
        if prompt is None and messages is None:
            raise ValueError("Must provide either prompt or messages.")

        self._output_parser = output_parser
        self._llm = llm
        self._prompt = prompt
        self._verbose = verbose
        self._messages = messages

        if self._prompt is not None:
            self._prompt.output_parser = output_parser

    @classmethod
    def from_defaults(
        cls,
        output_parser: PydanticOutputParser,
        prompt_template_str: Optional[str] = None,
        messages: Optional[List[ChatMessage]] = None,
        prompt: Optional[PromptTemplate] = None,
        llm: Optional[LLM] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "LLMTextCompletionProgram":
        llm = llm or OpenAI(temperature=0, model="gpt-3.5-turbo-0613")
        if prompt is None and prompt_template_str is None and messages is None:
            raise ValueError(
                "Must provide either prompt or prompt_template_str or messages."
            )
        if (
            prompt is not None
            and prompt_template_str is not None
            and messages is not None
        ):
            raise ValueError(
                "Must provide either prompt or prompt_template_str or messages."
            )
        if prompt_template_str is not None:
            prompt = PromptTemplate(prompt_template_str)

        return cls(
            output_parser,
            prompt=cast(PromptTemplate, prompt),
            messages=messages,
            llm=llm,
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
        response_fn = (
            self._llm.chat if self._llm.metadata.is_chat_model else self._llm.complete
        )
        formatted_arg = (
            self._prompt.format(**kwargs)
            if self._prompt is not None
            else self._messages
        )

        response = response_fn(formatted_arg)  # type: ignore
        raw_output = response.text
        model_output = self._output_parser.parse(raw_output)
        return model_output
