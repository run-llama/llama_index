from typing import Any, Optional, Type, cast

from llama_index.bridge.pydantic import BaseModel
from llama_index.llms.base import LLM
from llama_index.llms.openai import OpenAI
from llama_index.output_parsers.pydantic import PydanticOutputParser
from llama_index.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.types import BasePydanticProgram


class LLMTextCompletionProgram(BasePydanticProgram[BaseModel]):
    """
    LLM Text Completion Program.

    Uses generic LLM text completion + an output parser to generate a structured output.

    """

    def __init__(
        self,
        output_parser: PydanticOutputParser,
        prompt: BasePromptTemplate,
        llm: LLM,
        verbose: bool = False,
    ) -> None:
        self._output_parser = output_parser
        self._llm = llm
        self._prompt = prompt
        self._verbose = verbose

        self._prompt.output_parser = output_parser

    @classmethod
    def from_defaults(
        cls,
        output_parser: PydanticOutputParser,
        prompt_template_str: Optional[str] = None,
        prompt: Optional[PromptTemplate] = None,
        llm: Optional[LLM] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "LLMTextCompletionProgram":
        llm = llm or OpenAI(temperature=0, model="gpt-3.5-turbo-0613")
        if prompt is None and prompt_template_str is None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt is not None and prompt_template_str is not None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt_template_str is not None:
            prompt = PromptTemplate(prompt_template_str)
        return cls(
            output_parser,
            prompt=cast(PromptTemplate, prompt),
            llm=llm,
            verbose=verbose,
        )

    @property
    def output_cls(self) -> Type[BaseModel]:
        return self._output_parser.output_cls

    @property
    def prompt(self) -> BasePromptTemplate:
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: BasePromptTemplate) -> None:
        self._prompt = prompt

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> BaseModel:
        if self._llm.metadata.is_chat_model:
            messages = self._prompt.format_messages(llm=self._llm, **kwargs)

            response = self._llm.chat(messages)

            raw_output = response.message.content or ""
        else:
            formatted_prompt = self._prompt.format(llm=self._llm, **kwargs)

            response = self._llm.complete(formatted_prompt)

            raw_output = response.text

        return self._output_parser.parse(raw_output)

    async def acall(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> BaseModel:
        if self._llm.metadata.is_chat_model:
            messages = self._prompt.format_messages(llm=self._llm, **kwargs)

            response = await self._llm.achat(messages)

            raw_output = response.message.content or ""
        else:
            formatted_prompt = self._prompt.format(llm=self._llm, **kwargs)

            response = await self._llm.acomplete(formatted_prompt)

            raw_output = response.text

        return self._output_parser.parse(raw_output)
